import cv2
import numpy as np
from matplotlib import pyplot as plt

#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("test13.mp4")
ret, video = cam.read()
show_cam = np.copy(video)


#투상 변환 좌표 지정함수
selected_pts = [] #투상변환할 좌표를 배열로 저장하기 위한 변수
def mouse_callback(event, x, y, flags, param): # 화면에서 마우스 클릭시 호출되는 함수
    global selected_pts, show_cam

    if event == cv2.EVENT_LBUTTONUP:
        selected_pts.append([x, y])
        #cv2.circle(show_cam, (x,y),10,(0 ,255,0),3)

def select_points(image, points_num):  # 로드 되는 창에서 변환할 좌표 4 곳을 클릭
    global selected_pts
    selected_pts = []

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    while True:
        ret, video = cam.read()
        image = np.copy(video)
        if len(selected_pts) >= 1:        #점에 원 표시
            cv2.circle(image, tuple(selected_pts[0]), 10, (0, 0, 255), 3)
            if len(selected_pts) >= 2:
                cv2.circle(image, tuple(selected_pts[1]), 10, (0, 127, 255), 3)
                if len(selected_pts) >= 3:
                    cv2.circle(image, tuple(selected_pts[2]), 10, (0, 255, 255), 3)

        if ret == False:
            continue

        cv2.imshow('image', image)

        k = cv2.waitKey(1)
        if k == 27 or len(selected_pts) == points_num:
            break

    cv2.destroyAllWindows()
    return np.array(selected_pts, dtype=np.float32)

def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

show_cam = np.copy(video)
src_pts = select_points(show_cam, 4)
src_pts_yfix = np.array([src_pts[0],src_pts[1],[src_pts[2,0],src_pts[1,1]],[src_pts[3,0],src_pts[0,1]]], dtype=np.float32) #3, 4번 점의 y좌표를 각각 1, 2번 y좌표값과 같게 만듬
#src_pts_yfix = np.array([[src_pts[0,0],src_pts[0,1]],[src_pts[1,0],src_pts[1,1]],[src_pts[2,0],src_pts[1,1]],[src_pts[3,0],src_pts[0,1]]], dtype=np.float32)
print(src_pts_yfix)
dst_pts = np.array([[0, 400], [0, 0], [400, 0], [400, 400]], dtype=np.float32) #변환한 영상을 출력할 화면
#print(src_pts)

plt.ion()
while (True):
    ret, video = cam.read()

    #실시간 영상 일 때
    if ret == False:
       continue
    #동영상 일 때
    #    break

    show_cam = np.copy(video)
    perspective_m = cv2.getPerspectiveTransform(src_pts_yfix, dst_pts)
    unwarped_cam = cv2.warpPerspective(video, perspective_m, (400, 400)) #변환된 영상 출력 영역 (dst_pts 랑 똑같이 맞춰주면 됨)
    cv2.circle(show_cam, tuple(src_pts_yfix[0]), 10, (0, 0, 255), 3)   #변환하는 위치를 원본 이미지에 표시
    cv2.circle(show_cam, tuple(src_pts_yfix[1]), 10, (0, 127, 255), 3) #빨주노초, 왼쪽 아래부터 시계방향
    cv2.circle(show_cam, tuple(src_pts_yfix[2]), 10, (0, 255, 255), 3)
    cv2.circle(show_cam, tuple(src_pts_yfix[3]), 10, (0, 255, 0), 3)

    has_frame, frame = cam.read()
    if not has_frame:
        print('Reached the end of the video')
        break

    src = np.copy(unwarped_cam)

    # 1
    #    hsv    = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    #    h, s, v = cv2.split(hsv)
    #
    #    v2 = cv2.equalizeHist(v)
    #    hsv2 = cv2.merge([h, s, v2])
    #    dst    = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    #    cv2.imshow('dst',  dst)

    # 2
    yCrCv = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    y, Cr, Cv = cv2.split(yCrCv)

    y2 = np.copy(y)
    cv2.normalize(y, y2, 0, 255, cv2.NORM_MINMAX)
    yCrCv2 = cv2.merge([y2, Cr, Cv])
    img_smooth = cv2.cvtColor(yCrCv2, cv2.COLOR_YCrCb2BGR)

    plt.clf()

    hist1 = cv2.calcHist(images=[yCrCv], channels=[0], mask=None, histSize=[256 / 8], ranges=[0, 256])
    #    plt.subplot(1,2,1)
    plt.plot(hist1, color='b')
    hist2 = cv2.calcHist(images=[yCrCv2], channels=[0], mask=None, histSize=[256 / 8], ranges=[0, 256])
    #    plt.subplot(1,2,2)
    plt.plot(hist2, color='g')

    #resized_img = cv2.resize(video, (800, 600))
    #cv2.imshow('resized_img',resized_img)
#참고로 cam의 크기는 480, 640 이다. (세로, 가로)인 듯 cam.shape 로 cam의 크기 출력 가능
#resized_img = cv2.resize(image, (너비, 높이) ) 로 이미지 크기 고정 가능

    #이미지 hsv 해주는 코딩
    lower_yellow = (20, 100, 100)
    upper_yellow = (30, 255, 255)
    lower_white = (0, 0, 212)
    upper_white = (131, 255, 255)
    lower_blue = (110, 50, 50)
    upper_blue = (130, 255, 255)

    hsv_change = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2HSV)
    img_yellow_mask = cv2.inRange(hsv_change, lower_yellow, upper_yellow)
    hsv_yellow = cv2.bitwise_and(img_smooth , img_smooth , mask=img_yellow_mask)
    img_white_mask = cv2.inRange(hsv_change, lower_white, upper_white)
    hsv_white = cv2.bitwise_and(img_smooth, img_smooth, mask=img_white_mask)
    img_blue_mask = cv2.inRange(hsv_change, lower_blue, upper_blue)
    hsv_blue = cv2.bitwise_and(img_smooth, img_smooth, mask=img_blue_mask)
    temp = cv2.bitwise_or(hsv_yellow, hsv_white)
    img_hsv = cv2.bitwise_or(temp, hsv_blue)

    #이미지 모폴로지 실행
    gradient_change = cv2.resize(img_hsv, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    kernel = np.ones((11,11),np.uint8)
    img_morph = cv2.morphologyEx(img_hsv, cv2.MORPH_CLOSE, kernel)

    #캐니 에지
    img_canny = canny(img_morph, 70, 210)

    #허프 변환

    cv2.imshow('hsv', img_hsv)
    cv2.imshow('mark', img_smooth)
    cv2.imshow('canny', img_canny)
    #cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()