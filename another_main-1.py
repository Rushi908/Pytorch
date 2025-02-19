import roop.globals
from roop import utilities as util
import pathlib
from roop.ProcessEntry import ProcessEntry
import numpy as np
from roop.capturer import get_video_frame_total
from roop.face_util import extract_face_images
from roop.FaceSet import FaceSet
from roop.core import decode_execution_providers
from celery import Celery
import threading
from concurrent.futures import ThreadPoolExecutor

# Initialize Celery with Redis
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Celery Configuration
app.conf.update(
    task_acks_late=True,  # Prevents task loss if a worker crashes    worker_prefetch_multiplier=1,  # Each worker fetches only 1 task at a time
    broker_transport_options={'visibility_timeout': 1800},  # Allows tasks to stay in queue for 15 minutes
    worker_concurrency=2,  # At least 2 workers for simultaneous processing
)



list_files_process : list[ProcessEntry] = []

def search_facesets():
    SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path, (True, 0))
    
    for f in SELECTION_FACES_DATA:
        if not f:  # Check if `f` is empty
            print("Warning: No face detected in one of the entries. Skipping this entry.")
            continue

        face_set = FaceSet()
        face = f[0]
        face.mask_offsets = (0, 0, 0, 0, 1, 20)
        face_set.faces.append(face)
        roop.globals.INPUT_FACESETS.append(face_set)



def map_mask_engine(selected_mask_engine, clip_text):
    if selected_mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
          mask_engine = None
    elif selected_mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    else:
        mask_engine = None
    return mask_engine

 
@app.task(rate_limit='2/m') 
def start_swap(source_path, target_path, results, swap_model="InSwapper 128", output_method="File", enhancer="GFPGAN", detection="first", keep_frames=False, wait_after_extraction=False, skip_audio=False, face_distance=0.65, blend_ratio=0.65,
                selected_mask_engine=None, clip_text="cup, hands, hair, banana", processing_method="In-Memory processing", no_face_action="Use untouched original frame", vr_mode=False, autorotate=True, restore_original_mouth=False, num_swap_steps=1, upsample="128px"):
    
    # Handle list inputs from legacy calls
    if isinstance(source_path, list):
        source_path = source_path[0]
    if isinstance(target_path, list):
        target_path = target_path[0]
    if isinstance(results, list):
        results = results[0]

    from ui.main import prepare_environment
    from roop.core import batch_process_regular
        
    global is_processing, list_files_process

    roop.globals.execution_providers = decode_execution_providers(["cuda"])
    
    SELECTED_INPUT_FACE_INDEX = 0

    roop.globals.g_current_face_analysis = "one"
    roop.globals.g_desired_face_analysis = None

    # Validate path types
    if not all(isinstance(p, (str, pathlib.Path)) for p in [source_path, target_path, results]):
        raise ValueError("Invalid path types - expected string paths")

    roop.globals.source_path = str(source_path)
    roop.globals.target_path = str(target_path)
    roop.globals.output_path = str(results)

    prepare_environment(roop.globals.output_path)

    # Handle file extension safely
    target_path = pathlib.Path(target_path)
    if target_path.suffix:
        file_extension = target_path.suffix.lower().lstrip('.')
    else:
        file_extension = "mp4"  # Default fallback

    if file_extension in ["jpg", "png", "jpeg"]:
        roop.globals.output_image_format = file_extension
    else:
        roop.globals.output_video_format = file_extension

    search_facesets()

    # Rest of configuration remains the same
    roop.globals.selected_enhancer = enhancer
    roop.globals.cuda_device_id = 0
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.keep_frames = keep_frames
    roop.globals.wait_after_extraction = wait_after_extraction
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = detection
    roop.globals.no_face_action = no_face_action
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = autorotate
    roop.globals.subsample_size = int(upsample[:3])
    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    try:
        video_frames = util.detect_fps(str(target_path))
    except Exception as e:
        print(f"Error detecting FPS: {str(e)}")
        video_frames = 30  # Fallback to default FPS

    mask = {"layers": [np.zeros((600, 800, 4), dtype=np.uint8)]}

    is_processing = True            

    roop.globals.execution_threads = 2
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 14
    roop.globals.max_memory = None

    try:
        total_video_frames = get_video_frame_total(str(target_path))
    except Exception as e:
        print(f"Error getting frame total: {str(e)}")
        total_video_frames = 0

    process_entry = ProcessEntry(
        str(target_path),
        None,
        0,
        total_video_frames
    )
    process_entry.startframe = 0
    process_entry.fps = video_frames
    list_files_process.append(process_entry)

    batch_process_regular(
        swap_model,
        output_method,
        list_files_process,
        mask_engine,
        clip_text,
        processing_method == "In-Memory processing",
        mask,
        restore_original_mouth,
        num_swap_steps,
        SELECTED_INPUT_FACE_INDEX
    )
    
    is_processing = False
    outdir = pathlib.Path(roop.globals.output_path)
    outfiles = [str(item) for item in outdir.rglob("*") if item.is_file()]
    
    print(f"Processing complete. Output files: {outfiles}")
    return outfiles
    
def process_swap(input_path, target_path, results):
    """Runs the face swap process safely in a thread."""
    try:
        result = start_swap(input_path, target_path, results)
        print(f"Processing completed for: {input_path} with result: {result}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

@app.task(rate_limit='2/m')
def run_multithreading_swaps(input_paths, target_paths, results):
    """Executes multiple face swaps using optimized multithreading."""
    num_threads = 2  # At least 2 threads to handle parallel requests

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_swap, input_paths[i], target_paths[i], results) 
                   for i in range(len(input_paths))]

        for future in futures:
            future.result()  # Ensures all tasks complete before moving on

print("Celery worker ready for face swap tasks. Multi-user support enabled.")
