from imageai.Detection import VideoObjectDetection

vid_obj_detect = VideoObjectDetection()

vid_obj_detect.setModelTypeAsYOLOv3()


vid_obj_detect.setModelPath(r"yolo.h5")
vid_obj_detect.loadModel()


detected_vid_obj = vid_obj_detect.detectObjectsFromVideo(
    input_file_path =  r"input_video.mp4",
    output_file_path = r"output_video",
    frames_per_second=15,
    log_progress=True,
    return_detected_frame = True
)
