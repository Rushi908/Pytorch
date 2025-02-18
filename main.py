
import argparse
from another_main import start_swap,run_multithreading_swaps
import roop.globals

def main(input_paths, target_paths, results):
    try:
        print("Starting task...")
        result = run_multithreading_swaps.apply_async((input_paths, target_paths, results))
        print(f"Direct result: {result.get()}")
    except Exception as e:
        print(f"Error occurred: {e}")
        
if __name__ == "__main__":

    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))

    roop.globals.frame_processors = ['face_swapper', 'face_enhancer']

    program.add_argument("--source_paths", nargs="+", required=True)
    program.add_argument("--target_paths", nargs="+", required=True)
    program.add_argument("--results", nargs="+", required=True)

    roop.globals.startup_args = program.parse_args()

    args = program.parse_args()

    main(args.source_paths, args.target_paths, args.results)
    print("Face swap operations completed.")    
    
'''
from another_main import run_multithreading_swaps
import argparse


def main(input_paths, target_paths, results):
    try:
        print("Starting task...")
        result = run_multithreading_swaps.apply_async((input_paths, target_paths, results))
        print(f"Direct result: {result.get()}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Face Swap Script with Single GPU Support using Multithreading")
    
    # Arguments for input and output paths
    parser.add_argument('--input_paths', nargs='+', required=False, help="List of input image paths")
    parser.add_argument('--target_paths', nargs='+', required=False, help="List of target video paths")
    parser.add_argument('--results', required=False, help="Directory where results will be saved")
    parser.add_argument('--num_threads', type=int, default=4, help="Number of threads to use for multithreading")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # If command-line arguments are not provided, use default values inside the script
    # input_paths = args.input_paths if args.input_paths else ["F:\\face_swap_by_facefusionlib\\myra_photo.png"]
    # target_paths = args.target_paths if args.target_paths else ["F:\\face_swap_by_facefusionlib\\another_video_2.mp4"]
    # results = args.results if args.results else "F:\\face_swap_by_facefusionlib\\main_video_127.mp4"

    # Run the face swap operation
    main(args.input_paths, args.target_paths, args.results)
    print("Face swap operations completed.")
'''    