from moviepy.editor import ImageSequenceClip
import argparse
import os

IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    source_folder = os.path.join(args.image_folder, 'IMG/')
    
    #convert file folder into list firltered for image file types
    image_list = sorted([os.path.join(source_folder, image_file)
                        for image_file in os.listdir(source_folder)])
    
    image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT]

    print("Found", len(image_list), "files to create a video")
          
    #two methods of naming output video to handle varying environemnts
    video_file_1 = args.image_folder + '.mp4'
    video_file_2 = args.image_folder + 'output_video.mp4'

    print("Creating video from '{}', FPS={}".format(source_folder, args.fps))
    clip = ImageSequenceClip(image_list, fps=args.fps)
    
    encoding = 'mpeg4'
    
    try:
        clip.write_videofile(video_file_1, codec = encoding)
    except:
        clip.write_videofile(video_file_2, codec = encoding)


if __name__ == '__main__':
    main()
