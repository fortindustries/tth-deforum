''' infer.py for runpod worker '''

import os
import predict
from pathlib import Path
import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
import subprocess

MODEL = predict.Predictor()
MODEL.setup()

INPUT_SCHEMA = {
    'model_checkpoint': {
        'type': str,
        'required': False,
        'description': 'Choose stable diffusion model.',
        'default': 'deliberate-v2.safetensors',
    },
    'max_frames': {
        'type': int,
        'required': False,
        'description': 'Number of frames for animation',
        'default': 200,
    },
    'uncond_prompts': {
        'type': str,
        'required': False,
        'default': '',
    },
    'animation_prompts': {
        'type': str,
        'required': False,
        'description': "Prompt for animation. Provide 'frame number : prompt at this frame', separate different prompts with '|'. Make sure the frame number does not exceed the max_frames.",
        'default': '0: a beautiful apple, trending on Artstation | 50: a beautiful banana, trending on Artstation | 100: a beautiful coconut, trending on Artstation | 150: a beautiful durian, trending on Artstation',
    },
    'width': {
        'type': int,
        'required': False,
        'description': 'Width of output video. Reduce if out of memory.',
        'choices': [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
        'default': 512,
    },
    'height': {
        'type': int,
        'required': False,
        'description': 'Height of output image. Reduce if out of memory.',
        'choices': [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
        'default': 512,
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'description': 'Number of denoising steps',
        'ge': 1,
        'le': 500,
        'default': 50,
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'description': 'Scale for classifier-free guidance',
        'ge': 1,
        'le': 20,
        'default': 7,
    },
    'sampler': {
        'type': str,
        'required': False,
        'description': 'Choose a sampler.',
        'default': 'euler_ancestral',
        'choices': [
            'klms',
            'dpm2',
            'dpm2_ancestral',
            'heun',
            'euler',
            'euler_ancestral',
            'plms',
            'ddim',
            'dpm_fast',
            'dpm_adaptive',
            'dpmpp_2s_a',
            'dpmpp_2m',
        ],
    },
    'seed': {
        'type': int,
        'required': False,
        'description': 'Random seed. Leave blank to randomize the seed',
        'default': None,
    },
    'fps': {
        'type': int,
        'required': False,
        'description': 'Choose fps for the video.',
        'ge': 10,
        'le': 60,
        'default': 15,
    },
    'clip_name': {
        'type': str,
        'required': False,
        'description': 'Choose CLIP model',
        'default': 'ViT-L/14',
        'choices': ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32'],
        },
        'use_init': {
        'type': bool,
        'required': False,
        'default': False,
        'description': "If not using init image, you can skip the next settings to setting the animation_mode.",
        },
        'init_image': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'Provide init_image if use_init',
        },
        'use_mask': {
        'type': bool,
        'required': False,
        'default': False,
        },
        'mask_file': {
        'type': Path,
        'required': False,
        'default': None,
        'description': 'Provide mask_file if use_mask',
        },
        'invert_mask': {
        'type': bool,
        'required': False,
        'default': False,
        },
        'animation_mode': {
        'type': str,
        'required': False,
        'default': '2D',
        'choices': ['2D', '3D', 'Video Input', 'Interpolation'],
        'description': 'Choose Animation mode. All parameters below are for setting up animations.',
        },
        'border': {
        'type': str,
        'required': False,
        'default': 'replicate',
        'choices': ['wrap', 'replicate'],
        },
        'angle': {
        'type': str,
        'required': False,
        'default': '0:(0)',
        'description': 'angle parameter for the motion',
        },
        'zoom': {
        'type': str,
        'required': False,
        'default': '0:(1.04)',
        'description': 'zoom parameter for the motion',
        },
        'translation_x': {
        'type': str,
        'required': False,
        'default': '0:(10*sin(23.14*t/10))',
        'description': 'translation_x parameter for the 2D motion',
        },
        'translation_y': {
        'type': str,
        'required': False,
        'default': '0:(0)',
        'description': 'translation_y parameter for the 2D motion',
        },
        'translation_z': {
        'type': str,
        'required': False,
        'default': '0:(10)',
        'description': 'translation_z parameter for the 2D motion',
        },
        'rotation_3d_x': {
        'type': str,
        'required': False,
        'default': '0:(0)',
        'description': 'rotation_3d_x parameter for the 3D motion',
        },
        'rotation_3d_y': {
        'type': str,
        'required': False,
        'default': '0:(0)',
        'description': 'rotation_3d_y parameter for the 3D motion',
        },
        'rotation_3d_z': {
        'type': str,
        'required': False,
        'default': '0:(0)',
        'description': 'rotation_3d_z parameter for the 3D motion',
        },
        'flip_2d_perspective': {
            'type': bool,
            'required': False,
            'default': False,
        },
        'perspective_flip_theta': {
        'type': str,
        'required': False,
        'default': '0:(0)',
    },
    'perspective_flip_phi': {
        'type': str,
        'required': False,
        'default': '0:(t%15)',
    },
    'perspective_flip_gamma': {
        'type': str,
        'required': False,
        'default': '0:(0)',
    },
    'perspective_flip_fv': {
        'type': str,
        'required': False,
        'default': '0:(53)',
    },
    'noise_schedule': {
        'type': str,
        'required': False,
        'default': '0: (0.02)',
    },
    'strength_schedule': {
        'type': str,
        'required': False,
        'default': '0: (0.65)',
    },
    'contrast_schedule': {
        'type': str,
        'required': False,
        'default': '0: (1.0)',
    },
    'hybrid_comp_alpha_schedule': {
        'type': str,
        'required': False,
        'default': '0:(1)',
    },
    'hybrid_comp_mask_blend_alpha_schedule': {
        'type': str,
        'required': False,
        'default': '0:(0.5)',
    },
    'hybrid_comp_mask_contrast_schedule': {
        'type': str,
        'required': False,
        'default': '0:(1)',
    },
    'hybrid_comp_mask_auto_contrast_cutoff_high_schedule': {
        'type': str,
        'required': False,
        'default': '0:(100)',
    },
    'hybrid_comp_mask_auto_contrast_cutoff_low_schedule': {
        'type': str,
        'required': False,
        'default': '0:(0)',
    },
    'kernel_schedule': {
        'type': str,
        'required': False,
        'default': '0: (5)',
    },
    'sigma_schedule': {
        'type': str,
        'required': False,
        'default': '0: (1.0)',
    },
    'amount_schedule': {
        'type': str,
        'required': False,
        'default': '0: (0.2)',
    },
    'threshold_schedule': {
        'type': str,
        'required': False,
        'default': '0: (0.0)',
    },
    'color_coherence': {
        'type': str,
        'required': False,
        'default': 'Match Frame 0 LAB',
        'choices': ['Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'],
    },
    'color_coherence_video_every_N_frames': {
        'type': int,
        'required': False,
        'default': 1,
    },
    'diffusion_cadence': {
        'type': str,
        'required': False,
        'default': '1',
        'choices': ['1', '2', '3', '4', '5', '6', '7', '8'],
    },
    'use_depth_warping': {
        'type': bool,
        'required': False,
        'default': True,
    },
    'midas_weight': {
        'type': float,
        'required': False,
        'default': 0.3,
    },
    'near_plane': {
    'type': int,
    'required': False,
    'default': 200,
},
'far_plane': {
    'type': int,
    'required': False,
    'default': 10000,
},
'fov': {
    'type': int,
    'required': False,
    'default': 40,
},
'padding_mode': {
    'type': str,
    'required': False,
    'choices': ['border', 'reflection', 'zeros'],
    'default': 'border',
},
'sampling_mode': {
    'type': str,
    'required': False,
    'choices': ['bicubic', 'bilinear', 'nearest'],
    'default': 'bicubic',
},
'video_init_path': {
    'type': str,
    'required': False,
    'default': None,
},
'extract_nth_frame': {
    'type': int,
    'required': False,
    'default': 1,
},
'overwrite_extracted_frames': {
    'type': bool,
    'required': False,
    'default': True,
},
'use_mask_video': {
    'type': bool,
    'required': False,
    'default': False,
},
'video_mask_path': {
    'type': str,
    'required': False,
    'default': None,
},
'hybrid_generate_inputframes': {
    'type': bool,
    'required': False,
    'default': False,
},
'hybrid_use_first_frame_as_init_image': {
    'type': bool,
    'required': False,
    'default': True,
},
'hybrid_motion': {
    'type': str,
    'required': False,
    'choices': ['None', 'Optical Flow', 'Perspective', 'Affine'],
    'default': 'None',
},
'hybrid_flow_method': {
    'type': str,
    'required': False,
    'choices': ['Farneback', 'DenseRLOF', 'SF'],
    'default': 'Farneback',
},
'hybrid_composite': {
    'type': bool,
    'required': False,
    'default': False,
},
'hybrid_comp_mask_type': {
    'type': str,
    'required': False,
    'choices': ['None', 'Depth', 'Video Depth', 'Blend', 'Difference'],
    'default': 'None',
},
'hybrid_comp_mask_inverse': {
    'type': bool,
    'required': False,
    'default': False,
},
'hybrid_comp_mask_equalize': {
    'type': str,
    'required': False,
    'choices': ['None', 'Before', 'After', 'Both'],
    'default': 'None',
},
'hybrid_comp_mask_auto_contrast': {
    'type': bool,
    'required': False,
    'default': False,
},
'hybrid_comp_save_extra_frames': {
    'type': bool,
    'required': False,
    'default': False,
},
'hybrid_use_video_as_mse_image': {
    'type': bool,
    'required': False,
    'default': False,
},
'interpolate_key_frames': {
    'type': bool,
    'required': False,
    'default': False,
},
'interpolate_x_frames': {
    'type': int,
    'required': False,
    'default': 4,
},
    'resume_from_timestring': {
        'type': bool,
        'required': False,
        'default': False,
        },

    'resume_timestring': {
        'type': str,
        'required': False,
        'default': '',
    },


    'enable_schedule_samplers': {
        'type': bool,
        'required': False,
        'default': False,
    },


    'sampler_schedule:': {
        'type': str,
        'required': False,
        'default': '0: (0)',
    },
    }


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    MODEL.NSFW = job_input.get('nsfw', False)

    if validated_input['seed'] is None:
        validated_input['seed'] = int.from_bytes(os.urandom(2), "big")

    if validated_input['init_image'] != None:
        validated_input['use_init'] = True
        validated_input['init_image'] = Path(validated_input['init_image'])

    output_video_path = MODEL.predict(
        model_checkpoint=validated_input["model_checkpoint"],
        max_frames=validated_input['max_frames'],
        animation_prompts=validated_input['animation_prompts'],
        uncond_prompts=validated_input['uncond_prompts'],
        width=validated_input['width'],
        height=validated_input['height'],
        num_inference_steps=validated_input['num_inference_steps'],
        guidance_scale=validated_input['guidance_scale'],
        sampler=validated_input['sampler'],
        seed=validated_input['seed'],
        fps=validated_input['fps'],
        clip_name=validated_input['clip_name'],
        use_init=validated_input['use_init'],
        init_image=validated_input['init_image'],
        use_mask=validated_input['use_mask'],
        mask_file=validated_input['mask_file'],
        invert_mask=validated_input['invert_mask'],
        animation_mode=validated_input['animation_mode'],
        border=validated_input['border'],
        angle=validated_input['angle'],
        zoom=validated_input['zoom'],
        translation_x=validated_input['translation_x'],
        translation_y=validated_input['translation_y'],
        translation_z=validated_input['translation_z'],
        rotation_3d_x=validated_input['rotation_3d_x'],
        rotation_3d_y=validated_input['rotation_3d_y'],
        rotation_3d_z=validated_input['rotation_3d_z'],
        flip_2d_perspective=validated_input['flip_2d_perspective'],
        perspective_flip_theta=validated_input['perspective_flip_theta'],
        perspective_flip_phi=validated_input['perspective_flip_phi'],
        perspective_flip_gamma=validated_input['perspective_flip_gamma'],
        perspective_flip_fv=validated_input['perspective_flip_fv'],
        noise_schedule=validated_input['noise_schedule'],
        strength_schedule=validated_input['strength_schedule'],
        contrast_schedule=validated_input['contrast_schedule'],
        hybrid_comp_alpha_schedule=validated_input['hybrid_comp_alpha_schedule'],
        hybrid_comp_mask_blend_alpha_schedule=validated_input['hybrid_comp_mask_blend_alpha_schedule'],
        hybrid_comp_mask_contrast_schedule=validated_input['hybrid_comp_mask_contrast_schedule'],
        hybrid_comp_mask_auto_contrast_cutoff_high_schedule=validated_input['hybrid_comp_mask_auto_contrast_cutoff_high_schedule'],
        hybrid_comp_mask_auto_contrast_cutoff_low_schedule=validated_input['hybrid_comp_mask_auto_contrast_cutoff_low_schedule'],
        kernel_schedule=validated_input['kernel_schedule'],
        sigma_schedule=validated_input['sigma_schedule'],
        amount_schedule=validated_input['amount_schedule'],
        threshold_schedule=validated_input['threshold_schedule'],
        color_coherence=validated_input['color_coherence'],
        color_coherence_video_every_N_frames=validated_input['color_coherence_video_every_N_frames'],
        diffusion_cadence=validated_input['diffusion_cadence'],
        use_depth_warping=validated_input['use_depth_warping'],
        midas_weight=validated_input['midas_weight'],
        near_plane=validated_input['near_plane'],
        far_plane=validated_input['far_plane'],
        fov=validated_input['fov'],
        padding_mode=validated_input['padding_mode'],
        sampling_mode=validated_input['sampling_mode'],
        video_init_path=validated_input['video_init_path'],
        extract_nth_frame=validated_input['extract_nth_frame'],
        overwrite_extracted_frames=validated_input['overwrite_extracted_frames'],
        use_mask_video=validated_input['use_mask_video'],
        video_mask_path=validated_input['video_mask_path'],
        hybrid_generate_inputframes=validated_input['hybrid_generate_inputframes'],
        hybrid_use_first_frame_as_init_image=validated_input['hybrid_use_first_frame_as_init_image'],
        hybrid_motion=validated_input['hybrid_motion'],
        hybrid_flow_method=validated_input['hybrid_flow_method'],
        hybrid_composite=validated_input['hybrid_composite'],
        hybrid_comp_mask_type=validated_input['hybrid_comp_mask_type'],
        hybrid_comp_mask_inverse=validated_input['hybrid_comp_mask_inverse'],
        hybrid_comp_mask_equalize=validated_input['hybrid_comp_mask_equalize'],
        hybrid_comp_mask_auto_contrast=validated_input['hybrid_comp_mask_auto_contrast'],
        hybrid_comp_save_extra_frames=validated_input['hybrid_comp_save_extra_frames'],
        hybrid_use_video_as_mse_image=validated_input['hybrid_use_video_as_mse_image'],
        interpolate_key_frames=validated_input['interpolate_key_frames'],
        interpolate_x_frames=validated_input['interpolate_x_frames'],
        resume_from_timestring=validated_input['resume_from_timestring'],
        resume_timestring=validated_input['resume_timestring'],
        sampler_schedule=validated_input['resume_timestring'],
        enable_schedule_samplers=validated_input['enable_schedule_samplers']
    )

    job_output = {}

    bucket_creds = {}
    bucket_creds['bucketName'] = os.environ.get('BUCKET_NAME', None)
    bucket_creds['endpointUrl'] = os.environ.get('BUCKET_ENDPOINT_URL', None)
    bucket_creds['accessId'] = os.environ.get('BUCKET_ACCESS_KEY_ID', None)
    bucket_creds['accessSecret'] = os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)


    output_video_cropped_path = "/tmp/" + job['id'] + "-cropped.mp4"
    crop_video_center_ffmpeg(output_video_path, output_video_cropped_path)

    output_video_uploaded_url = rp_upload.file(job['id'], output_video_cropped_path,bucket_creds)




    job_output['video'] = output_video_uploaded_url
    job_output['seed'] = job_input['seed']
    job_output['refresh_worker'] = True

    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output


def crop_video_center_ffmpeg(input_path, output_path):
    original_width = 448
    original_height = 768
    new_width = 432
    new_height = 768

    # Calculate the crop values
    crop_left = (original_width - new_width) // 2
    crop_right = crop_left
    crop_top = 0
    crop_bottom = 0

    # Build the FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-filter:v", f"crop={new_width}:{new_height}:{crop_left}:{crop_top}",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]

    # Run the FFmpeg command
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, check=True)
    print(result.stderr)
    if result.returncode != 0:
        print("Error encountered:")
        print(result.stderr)

runpod.serverless.start({"handler": run})