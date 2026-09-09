# Grok Watermark Remover — Public V3-style Engine

This is the repo-ready public version of the locally proven Grok watermark-removal pipeline.

## Engine
- Fixed 720×1280 golden V3 binary mask.
- Mask scales to the uploaded video's dimensions.
- OpenCV `INPAINT_NS`, radius 3, on every decoded frame.
- H.264 output, CRF 18, `yuv420p`.
- Original FPS and dimensions preserved.
- Original audio stream copied with FFmpeg where present.
- 60 MB / 30 second safety limits for the free deployment.

## Deploy on Render Free
1. Put this folder in a Git repository.
2. In Render, create a new Web Service from that repo.
3. Select Docker and Free plan. Render supports Python/Docker web services and free web services can use custom domains; free services spin down after 15 minutes of inactivity.
4. After deployment, test with the golden reference video before attaching your custom domain.
5. Add the custom domain in Render when the processing test is confirmed.

## Important
This service uses ephemeral storage only. Uploaded and cleaned videos are temporary and are not intentionally persisted as a database or file library.
