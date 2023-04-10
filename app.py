from potassium import Potassium, Request, Response, send_webhook

from transformers import pipeline
import torch

import os
import predict
from pathlib import Path

app = Potassium("my_app")


MODEL = predict.Predictor()


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    MODEL.setup()

    context = {
        "model": MODEL
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    return Response(
        json={"outputs": outputs},
        status=200
    )


@app.async_handler("/async")
def async_handler(context: dict, request: Request) -> Response:

    job_input = request.json.get("input")

    MODEL.NSFW = job_input.get('nsfw', False)

    if input['seed'] is None:
        input['seed'] = int.from_bytes(os.urandom(2), "big")

    output_video_path = MODEL.predict(
        model_checkpoint=input["model_checkpoint"],
        max_frames=input['max_frames'],
        animation_prompts=input['animation_prompts'],
        width=input['width'],
        height=input['height'],
        num_inference_steps=input['num_inference_steps'],
        guidance_scale=input['guidance_scale'],
        sampler=input['sampler'],
        seed=input['seed'],
        fps=input['fps'],
        clip_name=input['clip_name'],
        use_init=input['use_init'],
        init_image=input['init_image'],
        use_mask=input['use_mask'],
        mask_file=input['mask_file'],
        invert_mask=input['invert_mask'],
        animation_mode=input['animation_mode'],
        border=input['border'],
        angle=input['angle'],
        zoom=input['zoom'],
        translation_x=input['translation_x'],
        translation_y=input['translation_y'],
        translation_z=input['translation_z'],
        rotation_3d_x=input['rotation_3d_x'],
        rotation_3d_y=input['rotation_3d_y'],
        rotation_3d_z=input['rotation_3d_z'],
        flip_2d_perspective=input['flip_2d_perspective'],
        perspective_flip_theta=input['perspective_flip_theta'],
        perspective_flip_phi=input['perspective_flip_phi'],
        perspective_flip_gamma=input['perspective_flip_gamma'],
        perspective_flip_fv=input['perspective_flip_fv'],
        noise_schedule=input['noise_schedule'],
        strength_schedule=input['strength_schedule'],
        contrast_schedule=input['contrast_schedule'],
        hybrid_video_comp_alpha_schedule=input['hybrid_video_comp_alpha_schedule'],
        hybrid_video_comp_mask_blend_alpha_schedule=input['hybrid_video_comp_mask_blend_alpha_schedule'],
        hybrid_video_comp_mask_contrast_schedule=input['hybrid_video_comp_mask_contrast_schedule'],
        hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule=input['hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule'],
        hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule=input['hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule'],
        kernel_schedule=input['kernel_schedule'],
        sigma_schedule=input['sigma_schedule'],
        amount_schedule=input['amount_schedule'],
        threshold_schedule=input['threshold_schedule'],
        color_coherence=input['color_coherence'],
        color_coherence_video_every_N_frames=input['color_coherence_video_every_N_frames'],
        diffusion_cadence=input['diffusion_cadence'],
        use_depth_warping=input['use_depth_warping'],
        midas_weight=input['midas_weight'],
        near_plane=input['near_plane'],
        far_plane=input['far_plane'],
        fov=input['fov'],
        padding_mode=input['padding_mode'],
        sampling_mode=input['sampling_mode'],
        video_init_path=input['video_init_path'],
        extract_nth_frame=input['extract_nth_frame'],
        overwrite_extracted_frames=input['overwrite_extracted_frames'],
        use_mask_video=input['use_mask_video'],
        video_mask_path=input['video_mask_path'],
        hybrid_video_generate_inputframes=input['hybrid_video_generate_inputframes'],
        hybrid_video_use_first_frame_as_init_image=input['hybrid_video_use_first_frame_as_init_image'],
        hybrid_video_motion=input['hybrid_video_motion'],
        hybrid_video_flow_method=input['hybrid_video_flow_method'],
        hybrid_video_composite=input['hybrid_video_composite'],
        hybrid_video_comp_mask_type=input['hybrid_video_comp_mask_type'],
        hybrid_video_comp_mask_inverse=input['hybrid_video_comp_mask_inverse'],
        hybrid_video_comp_mask_equalize=input['hybrid_video_comp_mask_equalize'],
        hybrid_video_comp_mask_auto_contrast=input['hybrid_video_comp_mask_auto_contrast'],
        hybrid_video_comp_save_extra_frames=input['hybrid_video_comp_save_extra_frames'],
        hybrid_video_use_video_as_mse_image=input['hybrid_video_use_video_as_mse_image'],
        interpolate_key_frames=input['interpolate_key_frames'],
        interpolate_x_frames=input['interpolate_x_frames'],
        resume_from_timestring=input['resume_from_timestring'],
        resume_timestring=input['resume_timestring']
    )

    job_output = {}

    bucket_creds = {}
    # bucket_creds['bucketName'] = 'canova-openjourney'
    # bucket_creds['endpointUrl'] = os.environ.get('BUCKET_ENDPOINT_URL', None)
    # bucket_creds['accessId'] = os.environ.get('BUCKET_ACCESS_KEY_ID', None)
    # bucket_creds['accessSecret'] = os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)

    # output_video_uploaded_url = rp_upload.file(job['id'], output_video_path,bucket_creds)

    job_output['video'] = str(output_video_path)
    job_output['seed'] = job_input['seed']
    job_output['refresh_worker'] = True

    send_webhook(url="http://localhost:8001", json={"outputs": job_output})

    return


if __name__ == "__main__":
    app.serve()
