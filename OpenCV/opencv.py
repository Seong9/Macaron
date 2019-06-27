import cv2
import numpy as np
from   matplotlib import pyplot as plt

ones_shape = [[0],[0],[0]]

#cam = cv2.VideoCapture('test13.mp4') #'C:/Users/USER/PycharmProjects/line_detect/kcity1'
cam = cv2.VideoCapture(1)
ret, video = cam.read()
video_copy1 = np.copy(video)

print(video.shape)
fps = cam.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
filename = 'line_detect.avi'
out = cv2.VideoWriter(filename, fourcc, fps, (852, 480))


selected_pts = [] #투상변환할 좌표를 배열로 저장하기 위한 변수
def mouse_callback(event, x, y, flags, param): # 화면에서 마우스 클릭시 호출되는 함수
    global selected_pts, video_copy1

    if event == cv2.EVENT_LBUTTONUP:
        selected_pts.append([x, y])
        #cv2.circle(video_copy1, (x,y),10,(0,255,0),3)

def select_points(image, points_num):  # 로드 되는 창에서 변환할 좌표 4 곳을 클릭
    global selected_pts
    selected_pts = []

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    while True:
        if len(selected_pts) >= 1:        #점에 원 표시
            cv2.circle(image, tuple(selected_pts[0]), 10, (0, 0, 255), 3)
            if len(selected_pts) >= 2:
                cv2.circle(image, tuple(selected_pts[1]), 10, (0, 127, 255), 3)
                if len(selected_pts) >= 3:
                    cv2.circle(image, tuple(selected_pts[2]), 10, (0, 255, 255), 3)

        cv2.imshow('image', image)

        k = cv2.waitKey(1)
        if k == 27 or len(selected_pts) == points_num:
            break

    cv2.destroyAllWindows()
    return np.array(selected_pts, dtype=np.float32)

src_pts = select_points(video_copy1, 4)
src_pts_yfix = np.array([src_pts[0],src_pts[1],[src_pts[2,0],src_pts[1,1]],[src_pts[3,0],src_pts[0,1]]], dtype=np.float32) #3, 4번 점의 y좌표를 각각 1, 2번 y좌표값과 같게 만듬
print(src_pts_yfix)

dst_pts = np.array([[0, 600], [0, 0], [400, 0], [400, 600]], dtype=np.float32) #변환한 영상을 출력할 화면
#dst_pts = np.array([[150, 600], [0, 0], [400, 0], [250, 600]], dtype=np.float32) #전시야 새눈뷰


plt.ion()
while True:
    ret, video = cam.read()

    if ret == False:
        break

    video_copy1 = np.copy(video) #편집용 video copy1
    video_copy2 = np.copy(video) #필터용 copy2

    video_copy2_blur = cv2.GaussianBlur(video_copy2, ksize=(3, 3), sigmaX=0.0) #필터적용(블러) copy3
    video_copy2_blur_copy1 = np.copy(video_copy2_blur) #필터적용copy

    yCrCv = cv2.cvtColor(video_copy2_blur_copy1, cv2.COLOR_BGR2YCrCb)
    y, Cr, Cv = cv2.split(yCrCv)

    y2 = np.copy(y)
    cv2.normalize(y, y2, 0, 255, cv2.NORM_MINMAX)  # 정규화
    yCrCv2 = cv2.merge([y2, Cr, Cv])
    video_copy2_blur_copy1_normalize = cv2.cvtColor(yCrCv2, cv2.COLOR_YCrCb2BGR)
    video_copy2_blur_copy1_normalize_copy1 = np.copy(video_copy2_blur_copy1_normalize)
    video_copy2_blur_copy1_normalize_copy2 = np.copy(video_copy2_blur_copy1_normalize)

    #cv2.imshow('video', video)  # 원본 이미지 출력
    cv2.imshow('video_copy2_blur_copy1_normalize', video_copy2_blur_copy1_normalize)  # 블러 + 정규화 이미지 출력

    perspective_m = cv2.getPerspectiveTransform(src_pts_yfix, dst_pts)

    cv2.circle(video_copy2_blur_copy1_normalize_copy2, tuple(src_pts_yfix[0]), 10, (0, 0, 255), 3)  # 변환하는 위치를 원본 이미지에 표시
    cv2.circle(video_copy2_blur_copy1_normalize_copy2, tuple(src_pts_yfix[1]), 10, (0, 127, 255), 3)  # 빨주노초, 왼쪽 아래부터 시계방향
    cv2.circle(video_copy2_blur_copy1_normalize_copy2, tuple(src_pts_yfix[2]), 10, (0, 255, 255), 3)
    cv2.circle(video_copy2_blur_copy1_normalize_copy2, tuple(src_pts_yfix[3]), 10, (0, 255, 0), 3)

    video_copy2_blur_copy1_normalize_copy1_PTtrans = cv2.warpPerspective(video_copy2_blur_copy1_normalize_copy1, perspective_m, (400, 600))  # 변환된 영상 출력 영역 (dst_pts 랑 똑같이 맞춰주면 됨)
    video_copy2_blur_copy1_normalize_copy2_PTtrans_circle = cv2.warpPerspective(video_copy2_blur_copy1_normalize_copy2, perspective_m, (400, 600))  # circle 포함

    #    cv2.imshow('result', video_copy2_blur_copy1_normalize_copy_PTtrans)  # 블러 + 정규화 + 변환 이미지 출력
    #    cv2.imshow('result', video_copy2_blur_copy1_normalize_copy_PTtrans_circle)  # 블러 + 정규화 + 변환 + circle 이미지 출력

    lower_yellow = (20, 100, 100)
    upper_yellow = (30, 255, 255)
    lower_blue = (80, 50, 180)
    upper_blue = (115, 255, 255)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡHSV Blue Yellow 추출ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    video_hsv = np.copy(video_copy2_blur_copy1_normalize_copy1)
#    video_hsv = np.copy(video_copy2_blur_copy1_normalize_copy1_PTtrans)
    hsv_change = cv2.cvtColor(video_hsv, cv2.COLOR_BGR2HSV)
    img_yellow_mask = cv2.inRange(hsv_change, lower_yellow, upper_yellow)
    bgr_yellow_detect = cv2.bitwise_and(video_hsv, video_hsv, mask=img_yellow_mask)

    img_blue_mask = cv2.inRange(hsv_change, lower_blue, upper_blue)
#    bgr_blue_detect = cv2.bitwise_and(video_hsv, video_hsv, mask=img_blue_mask)
    white = np.full(hsv_change.shape, 255, dtype=np.float32)
    blue_area_detect = cv2.bitwise_and(white, white, mask=img_blue_mask)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
    opening_blue = cv2.morphologyEx(blue_area_detect, cv2.MORPH_OPEN, kernel, iterations=2)
    closing_blue = cv2.morphologyEx(opening_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_area_detect_morph = np.copy(opening_blue)
    blue_area_detect_morph_hsv = cv2.cvtColor(blue_area_detect_morph, cv2.COLOR_BGR2HSV)
    img_blue_mask_morph = cv2.inRange(blue_area_detect_morph_hsv, (0,0,212), (131,255,255))
    brg_blue_detect_morph = cv2.bitwise_and(video_hsv, video_hsv, mask=img_blue_mask_morph)


    temp = cv2.bitwise_or(bgr_yellow_detect, brg_blue_detect_morph)
    temp_hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡHSV Blue Yellow 추출ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ



#    print(bgr_blue_detect.shape)




#    cv2.imshow('brg_blue_detect_morph', brg_blue_detect_morph)


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡHSV lower upper 값 추적용 히스토그램ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    brg_blue_detect_morph_copy = np.copy(brg_blue_detect_morph)
    brg_blue_detect_morph_copy_hsv = cv2.cvtColor(brg_blue_detect_morph_copy, cv2.COLOR_BGR2HSV)
    plt.clf()
    histColor = ('b', 'g', 'r')
    binX = np.arange(32)*8
    plt.ylim(0,4000)
    for i in range(3):
        hist = cv2.calcHist(images=[brg_blue_detect_morph_copy_hsv], channels=[i], mask=None, histSize=[256 / 8], ranges=[0, 256])
        plt.plot(binX, hist, color=histColor[i])
    plt.show()
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡHSV lower upper 값 추적용 히스토그램ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이진화ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    video_binary = np.copy(video_copy2_blur_copy1_normalize_copy1)
    gray_change = cv2.cvtColor(video_binary, cv2.COLOR_BGR2GRAY)
    ret_b, binary_detect = cv2.threshold(gray_change, 210, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
    closing_binary = cv2.morphologyEx(binary_detect, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening_binary = cv2.morphologyEx(closing_binary, cv2.MORPH_OPEN, kernel, iterations=7)

    binary_detect_morph = np.copy(closing_binary)
    binary_detect_morph_brg = cv2.cvtColor(binary_detect_morph, cv2.COLOR_GRAY2BGR)
    #cv2.imshow('binary_detect_morph', binary_detect_morph)
    #cv2.imshow('gray_change', gray_change)
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이진화ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


    add_image = cv2.add(brg_blue_detect_morph, binary_detect_morph_brg)
    cv2.imshow('add_image',add_image)
    out.write(add_image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

