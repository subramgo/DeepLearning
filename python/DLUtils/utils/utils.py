"""
	Adhoc utitliy functions

"""
import subprocess

def video2images(in_video_path, out_image_path):
	""" Extract frames from  videos and save them"""
	print("Creating images from video {}".format(in_video_path))
	ffmpeg_cmd = "ffmpeg -i " + in_video_path +" -f image2 " + out_image_path + "/video-frame%05d.jpeg"
	subprocess.call(ffmpeg_cmd , shell=True)
	print("Image extraction complete. Images in folder {}".format(out_image_path))





if __name__ == '__main__':
	video2images('/Users/gsubramanian/Downloads/Walter/Galaxis/Galaxis-NVR1_11_59_57_30_1_2018.mp4',
		'/Users/gsubramanian/Downloads/Walter/Galaxis/images')




