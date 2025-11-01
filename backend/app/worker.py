import asyncio
import json
import logging
import os
import shlex
import time
from typing import Any, Dict, List, Optional

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

from .pipeline import (
    _annotate_segments,
    _extract_tracks,
    _whisper_transcribe,
    mux_stage,
    translate_stage,
    tts_finalize_stage,
)
from .utils import ffprobe_duration, mask_keep_intervals, run
from .utils_meta import load_meta, save_meta
from .vad import complement_intervals, compute_vad_silences, merge_intervals, sum_silence_between

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("WORKER_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


class JobProcessingError(Exception):
    """Raised when a job fails irrecoverably during processing."""


def _ensure_workdir(job_id: str) -> str:
    workdir = os.path.join("/app/data", job_id)
    os.makedirs(workdir, exist_ok=True)
    return workdir


def _run_asr_stage(job_id: str, input_path: str) -> Dict[str, Any]:
    """
    Execute ASR pipeline for a given job and persist metadata.
    """
    workdir = _ensure_workdir(job_id)
    full_48k, vocals_48k, bgm_48k, vocals_16k_raw = _extract_tracks(input_path, workdir)
    total = ffprobe_duration(full_48k)

    silences = compute_vad_silences(
        vocals_16k_raw,
        aggressiveness=int(os.getenv("VAD_AGGR", "3")),
        frame_ms=int(os.getenv("VAD_FRAME_MS", "30")),
    )

    segments = _whisper_transcribe(vocals_16k_raw)

    margin = float(os.getenv("STT_INTERVAL_MARGIN", "0.10"))
    stt_intervals = merge_intervals(
        [
            (
                max(0.0, float(s["start"]) - margin),
                min(float(total), float(s["end"]) + margin),
            )
            for s in segments
            if float(s["end"]) > float(s["start"])
        ]
    )

    speech_only_48k = os.path.join(workdir, "speech_only_48k.wav")
    vocals_fx_48k = os.path.join(workdir, "vocals_fx_48k.wav")
    mask_keep_intervals(vocals_48k, stt_intervals, speech_only_48k, sr=48000, ac=2)
    nonspeech_intervals = complement_intervals(stt_intervals, total)
    mask_keep_intervals(vocals_48k, nonspeech_intervals, vocals_fx_48k, sr=48000, ac=2)

    wav_16k = os.path.join(workdir, "speech_16k.wav")
    run(
        f"ffmpeg -y -i {shlex.quote(speech_only_48k)} -ac 1 -ar 16000 -c:a pcm_s16le {shlex.quote(wav_16k)}"
    )

    for i in range(len(segments)):
        if i < len(segments) - 1:
            st = float(segments[i]["end"])
            en = float(segments[i + 1]["start"])
            segments[i]["gap_after_vad"] = sum_silence_between(silences, st, en)
            segments[i]["gap_after"] = max(0.0, en - st)
        else:
            segments[i]["gap_after_vad"] = 0.0
            segments[i]["gap_after"] = 0.0

    _annotate_segments(segments)

    meta = {
        "job_id": job_id,
        "workdir": workdir,
        "input": input_path,
        "audio_full_48k": full_48k,
        "vocals_48k": vocals_48k,
        "bgm_48k": bgm_48k,
        "speech_only_48k": speech_only_48k,
        "vocals_fx_48k": vocals_fx_48k,
        "wav_16k": wav_16k,
        "orig_duration": total,
        "segments": segments,
        "silences": silences,
        "speech_intervals_stt": stt_intervals,
        "nonspeech_intervals_stt": nonspeech_intervals,
    }
    save_meta(workdir, meta)
    return meta


def _build_segment_payload(meta: Dict[str, Any], translations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    segs = meta.get("segments") or []
    for idx, seg in enumerate(segs):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        length = seg.get("length")
        if length is None:
            length = max(0.0, end - start)
        translation_text = ""
        if idx < len(translations):
            translation_text = translations[idx].get("text", "")
        payload.append(
            {
                "seg_id": seg.get("seg_id", idx),
                "seg_txt": seg.get("text", ""),
                "score": seg.get("score"),
                "start": start,
                "end": end,
                "editor": None,
                "trans_txt": translation_text,
                "length": length,
                "issues": seg.get("issues", []),
            }
        )
    return payload


class QueueWorker:
    def __init__(self) -> None:
        self.queue_url = os.environ["JOB_QUEUE_URL"]
        self.bucket = os.environ["AWS_S3_BUCKET"]
        self.region = os.getenv("AWS_REGION", "ap-northeast-2")
        self.default_target_lang = os.getenv("JOB_TARGET_LANG", "en")
        self.default_source_lang = os.getenv("JOB_SOURCE_LANG", "ko")
        self.result_video_prefix = os.getenv(
            "JOB_RESULT_VIDEO_PREFIX", "projects/{project_id}/outputs/videos/{job_id}.mp4"
        )
        self.result_meta_prefix = os.getenv(
            "JOB_RESULT_METADATA_PREFIX", "projects/{project_id}/outputs/metadata/{job_id}.json"
        )
        self.visibility_timeout = int(os.getenv("JOB_VISIBILITY_TIMEOUT", "300"))
        self.poll_wait = int(os.getenv("JOB_QUEUE_WAIT", "20"))

        session_kwargs: Dict[str, Any] = {}
        profile = os.getenv("AWS_PROFILE")
        if profile:
            session_kwargs["profile_name"] = profile
        self.boto_session = boto3.Session(region_name=self.region, **session_kwargs)
        self.sqs = self.boto_session.client("sqs", region_name=self.region)
        self.s3 = self.boto_session.client("s3", region_name=self.region)
        self.http = requests.Session()

    def poll_forever(self) -> None:
        logger.info("Starting SQS poller for queue %s", self.queue_url)
        while True:
            try:
                messages = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=self.poll_wait,
                    VisibilityTimeout=self.visibility_timeout,
                    MessageAttributeNames=["All"],
                )
            except (BotoCoreError, ClientError) as exc:
                logger.error("Failed to poll SQS: %s", exc)
                time.sleep(5)
                continue

            for msg in messages.get("Messages", []):
                receipt = msg["ReceiptHandle"]
                try:
                    body = json.loads(msg.get("Body", "{}"))
                except json.JSONDecodeError:
                    logger.error("Invalid message body, deleting: %s", msg.get("Body"))
                    self._delete_message(receipt)
                    continue

                success = False
                try:
                    self._handle_job(body)
                    success = True
                except JobProcessingError as exc:
                    logger.error("Job %s failed: %s", body.get("job_id"), exc)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("Unexpected error handling message: %s", exc)

                if success:
                    self._delete_message(receipt)

    def _delete_message(self, receipt: str) -> None:
        try:
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt)
        except (BotoCoreError, ClientError) as exc:
            logger.error("Failed to delete SQS message: %s", exc)

    def _handle_job(self, payload: Dict[str, Any]) -> None:
        job_id = payload.get("job_id")
        project_id = payload.get("project_id")
        input_key = payload.get("input_key")
        callback_url = payload.get("callback_url")
        if not all([job_id, project_id, input_key, callback_url]):
            raise JobProcessingError("Missing required job fields in payload")

        target_lang = payload.get("target_lang") or self.default_target_lang
        source_lang = payload.get("source_lang") or self.default_source_lang

        workdir = _ensure_workdir(job_id)
        extension = os.path.splitext(input_key)[1]
        local_input = os.path.join(workdir, f"input{extension or '.mp4'}")

        try:
            logger.info("Downloading s3://%s/%s to %s", self.bucket, input_key, local_input)
            self.s3.download_file(self.bucket, input_key, local_input)

            self._post_status(callback_url, "in_progress", metadata={"stage": "downloaded"})

            meta = _run_asr_stage(job_id, local_input)
            translations = translate_stage(meta["segments"], src=source_lang, tgt=target_lang)
            meta["translations"] = translations
            save_meta(workdir, meta)

            self._post_status(callback_url, "in_progress", metadata={"stage": "tts_prepare"})

            asyncio.run(tts_finalize_stage(job_id, target_lang, None))
            output_path = mux_stage(job_id)

            meta = load_meta(workdir)
            segments_payload = _build_segment_payload(meta, translations)

            result_key = self.result_video_prefix.format(project_id=project_id, job_id=job_id)
            metadata_key = self.result_meta_prefix.format(project_id=project_id, job_id=job_id)

            logger.info("Uploading result video to s3://%s/%s", self.bucket, result_key)
            self.s3.upload_file(output_path, self.bucket, result_key)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=metadata_key,
                Body=json.dumps(
                    {
                        "job_id": job_id,
                        "project_id": project_id,
                        "segments": segments_payload,
                        "target_lang": target_lang,
                        "source_lang": source_lang,
                        "input_key": input_key,
                        "result_key": result_key,
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                ContentType="application/json",
            )

            status_payload = {
                "stage": "completed",
                "segments_count": len(segments_payload),
                "segments": segments_payload,
                "metadata_key": metadata_key,
                "result_key": result_key,
                "target_lang": target_lang,
                "source_lang": source_lang,
                "input_key": input_key,
            }

            self._post_status(
                callback_url,
                "done",
                result_key=result_key,
                metadata=status_payload,
            )
        except (BotoCoreError, ClientError) as exc:
            failure = JobProcessingError(f"AWS client error: {exc}")
            self._safe_fail(callback_url, failure)
            raise failure
        except JobProcessingError as exc:
            self._safe_fail(callback_url, exc)
            raise
        except Exception as exc:  # pylint: disable=broad-except
            wrapped = JobProcessingError(str(exc))
            self._safe_fail(callback_url, wrapped)
            raise wrapped

    def _safe_fail(self, callback_url: str, error: JobProcessingError) -> None:
        try:
            self._post_status(
                callback_url,
                "failed",
                error=str(error),
                metadata={"stage": "failed"},
            )
        except JobProcessingError as callback_exc:
            logger.error("Failed to deliver failure callback: %s", callback_exc)

    def _post_status(
        self,
        callback_url: str,
        status: str,
        *,
        result_key: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {"status": status}
        if result_key is not None:
            payload["result_key"] = result_key
        if error is not None:
            payload["error"] = error
        if metadata is not None:
            payload["metadata"] = metadata

        try:
            resp = self.http.post(callback_url, json=payload, timeout=30)
        except requests.RequestException as exc:
            raise JobProcessingError(f"Callback request failed: {exc}") from exc

        if not resp.ok:
            raise JobProcessingError(
                f"Callback responded with {resp.status_code}: {resp.text[:200]}"
            )


def run_worker() -> None:
    worker = QueueWorker()
    worker.poll_forever()


if __name__ == "__main__":
    run_worker()
