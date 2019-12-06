# Imports
import numpy as np
import cv2
import math
import time

# 웹 카메라는 0번!
capture = cv2.VideoCapture(0)

#Window 이름 설정
test_title = "test"

# 'skin'의 범위 값 설정 
#HSV Color 찾기 
lower = np.array([2, 0, 0], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

count = 0
ave_time=0

while capture.isOpened():

    #비디오의 한 프레임씩 읽는다. 제대로 프레임을 읽으면 ret값이 True, 실패하면 False가 나타난다.
    ret, frame = capture.read()
    if not ret:
        break
        
    laptime = time.time()

    #화면 전체를 사용할 때 흰색이 아닌 것들이 나오면 화면 에러가 떠서 Detection Box를 만들어 그 부분만 DETECT 하게 만들었습니다.
    #[중간 이후] 전체 화면을 detect 하도록 만들었습니다
    #cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    detection_image = frame

    #지정된 프레임에 가우시안 블러적용해서 노이즈를 지운 부드러운 값을 사용
    #mask 만들 때 사용할 것
    #교수님 질문 GaussianBlur 0 에 대해서 설명하세요 
    #0의 의미는 자동으로 표준 편차 값 사용(221pg)
    blur = cv2.GaussianBlur(detection_image, (3, 3), 0)

    #BGR->HSV로 변환
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #lower upper 를 통해 살색으로 지정한 범위의 값에 대한 마스크 만들어주기(살색 범위는 하양 / 아니면 검은색)
    mask = cv2.inRange(hsv, lower, upper)

    #비어있는 array 만들어주기
    #1로 채워진 5*5 배열로 된 커널을 만들어 준다.
    #교수님 질문 (5,5)를 왜 하는가
    #1*1 < 3*3 < 5*5 로 갈수록 강도가 커진다. => 이론 참고 => https://webnautes.tistory.com/1257 and https://webnautes.tistory.com/1257
    kernel = np.ones((5, 5))

    #배경 잡음 제거[closing 순서와 같다]
    #팽창 => mask 의 흰색 구역에 대한 노이즈 제거 (좀 뚱뚱해짐)
    dilation = cv2.dilate(mask, kernel)
    #침식 => mask의 검은 구역에 대한 노이즈 제거
    #이미지의 경계부분을 침식시켜서 Background 이미지로 전환한다.
    erosion = cv2.erode(dilation, kernel)

    #한번 더 노이즈 처리 
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
    
    #cv2.imshow("Thresholded", thresh)

    #경계선(윤곽) 찾기
    #hierarchy = 잡힌 경계선을 계층으로 저장한다.(사용 x)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        #최대 면적을 가진 경계선 찾기
        #contours 에서 가장 넓은 면적을 가진 값을 contour로 설정
        #개념 => https://bit.ly/33Y5gJk  의 contours[0]보다는 업그레이드한 것
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        #윤곽선 주위에 경계가 되는 사각형 만들기
        #컨투어를 둘러싸는 박스는 boundingRect 함수로 구한다.
        #참고 = >https://datascienceschool.net/view-notebook/f9f8983941254a34bf0fee42c66c5539/
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(detection_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        #볼록 선체 찾기
        #개념 참고 => https://hoony-gunputer.tistory.com/103 및 전공책 265pg
        hull = cv2.convexHull(contour)
        
        #개념 참고 => https://leechamin.tistory.com/257?category=857057 및 258pg
        #볼록 선체의 넓이
        areahull = cv2.contourArea(hull)
        #컨투어의 넓이
        areacnt = cv2.contourArea(contour)
        
        #결과 보고서에 설명 첨부
        #나중에 90이하의 angle에서 0과1을 구분하는데 사용합니다.
        arearatio=((areahull-areacnt)/areacnt)*100

        #볼록 선체 및 컨투어 시각적 표시
        drawing = np.zeros(detection_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        #경계선의 오목한 부분
        #참고 => https://leechamin.tistory.com/259
        #returnPoints = False ==> hull points에 상응하는 contour points의 지표 반환
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        #코사인 규칙을 사용하여 시작점과 끝점에서 먼 지점의 각도(볼록한 지점)을 찾는다.
        #즉 손가락 끝점과 끝점 사이의 각도를 셀 변수
        count_defects = 0

        for i in range(defects.shape[0]):
            #시작 , 종료 , 가장 먼 지점 , 거리
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            #삼각형의 모든면의 길이를 구합니다
            #참고 => https://www.youtube.com/user/8vishwajeet 님의 detection 알고리즘 코드를 따라했습니다.
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # 각도가 90보다 작거나 크면 빨간색 원을 그림
            #손가락이 들어가는 부분 (오목한 부분)에 원을 그림
            if angle <= 90:
                count_defects += 1
                cv2.circle(detection_image, far, 1, [0, 0, 255], -1)
                
            #손으로 인식한 부분에 초록색 선 그림
            cv2.line(detection_image, start, end, [0, 255, 0], 2)

        #손가락 개수
        if count_defects == 0:
            if arearatio<12:
                #대기 상태
                #10에서 15사이여야 0과 1을 구분할 백분율이 나온다.
                cv2.putText(frame, "0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
                cv2.imshow(test_title, frame)
            else:
                #캐니 엣지
                #thresh 를 통해 배경의 엣지는 잡히지 않도록 만든다.
                cv2.putText(frame, "1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
                cv2.imshow(test_title, cv2.Canny(thresh,100,200))
        elif count_defects == 1:
            #스케치
            #영상 처리 교과서 예제 참고
            cv2.putText(frame, "2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            # 그레이 스케일로 변경    
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 잡음 제거를 위해 가우시안 플러 필터 적용(라플라시안 필터 적용 전에 필수)
            img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
            # 라플라시안 필터로 엣지 거출
            edges = cv2.Laplacian(img_gray, -1, None, 5)
            # 스레시홀드로 경계 값 만 남기고 제거하면서 화면 반전(흰 바탕 검은 선)
            ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

            # 경계선 강조를 위해 팽창 연산
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            sketch = cv2.erode(sketch, kernel)
            # 경계선 자연스럽게 하기 위해 미디언 블러 필터 적용
            sketch = cv2.medianBlur(sketch, 5)
            # 그레이 스케일에서 BGR 컬러 스케일로 변경
            img_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

            # 컬러 이미지 선명선을 없애기 위해 평균 블러 필터 적용
            img_paint = cv2.blur(frame, (10,10) )
            # 컬러 영상과 스케치 영상과 합성
            img_paint = cv2.bitwise_and(img_paint, img_paint, mask=sketch)

            # 결과 출력
            cv2.imshow(test_title, img_paint)
        elif count_defects == 2:
            #볼록 왜곡(선택)
            #영상 처리 교과서 예제 참고
            cv2.putText(frame, "3", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            rows,cols = detection_image.shape[:2]
            exp = 1.5
            scale = 1
            
            mapy, mapx = np.indices((rows, cols),dtype=np.float32)
            # 좌상단 기준좌표에서 -1~1로 정규화된 중심점 기준 좌표로 변경
            mapx = (2*mapx - cols)/cols
            mapy = (2*mapy - rows)/rows
            
            r,theta = cv2.cartToPolar(mapx,mapy)
            r[r< scale] = r[r<scale] **exp
            
            # 극 좌표를 직교좌표로 변환
            mapx, mapy = cv2.polarToCart(r, theta)
            
            # 중심점 기준에서 좌상단 기준으로 변경
            mapx = ((mapx + 1)*(cols-1))/2
            mapy = ((mapy + 1)*(rows-1))/2
            
            Convex_distorted = cv2.remap(detection_image,mapx,mapy,cv2.INTER_LINEAR)
            cv2.imshow(test_title,Convex_distorted)
        elif count_defects == 3:
            #오목 왜곡
            #영상 처리 교과서 예제 참고
            cv2.putText(frame, "4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            rows,cols = detection_image.shape[:2]
            exp = 0.5
            scale = 1
            
            mapy, mapx = np.indices((rows, cols),dtype=np.float32)
            # 좌상단 기준좌표에서 -1~1로 정규화된 중심점 기준 좌표로 변경
            mapx = (2*mapx - cols)/cols
            mapy = (2*mapy - rows)/rows
            
            r,theta = cv2.cartToPolar(mapx,mapy)
            r[r< scale] = r[r<scale] **exp
            
            # 극 좌표를 직교좌표로 변환
            mapx, mapy = cv2.polarToCart(r, theta)
            
            # 중심점 기준에서 좌상단 기준으로 변경
            mapx = ((mapx + 1)*(cols-1))/2
            mapy = ((mapy + 1)*(rows-1))/2
            
            Concave_distorted = cv2.remap(detection_image,mapx,mapy,cv2.INTER_LINEAR)
            cv2.imshow(test_title,Concave_distorted)
        elif count_defects == 4:
            #방사 왜곡(선택)
            #영상 처리 교과서 예제 참고
            cv2.putText(frame, "5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
            # 왜곡 계수 설정
            k1, k2, k3 = 0.5, 0.2, 0.0 # 배럴 왜곡
            rows, cols =  detection_image.shape[:2]
            
            # 매핑 배열 생성
            mapy, mapx = np.indices((rows, cols),dtype=np.float32)

            # 중앙점 좌표로 -1~1 정규화 및 극좌표 변환
            mapx = (2*mapx - cols)/cols
            mapy = (2*mapy - rows)/rows
            r, theta = cv2.cartToPolar(mapx, mapy)

            # 방사 왜곡 변형 연산
            ru = r*(1+k1*(r**2) + k2*(r**4) + k3*(r**6)) 

            # 직교좌표 및 좌상단 기준으로 복원
            mapx, mapy = cv2.polarToCart(ru, theta)
            mapx = ((mapx + 1)*cols)/2
            mapy = ((mapy + 1)*rows)/2

            # 리매핑
            distored = cv2.remap(detection_image,mapx,mapy,cv2.INTER_LINEAR)
            cv2.imshow(test_title,distored)
        else:
            pass
    except:
        pass

    #이건 중간 결과물 이미지로 손의 개수가 맞을 때만 특정 윈도우가 실행됩니다.
#     cv2.imshow('origin', frame)
#     cv2.imshow('edge', cv2.Canny(thresh,100,200))
#     cv2.imshow('sketch', img_paint)
#     cv2.imshow('convex',Convex_distorted)
#     cv2.imshow('concave',Concave_distorted)
#     cv2.imshow('distored',distored)

    laptime = time.time()-laptime
    count+=1
    ave_time =(ave_time*(count-1)+laptime)/count

    #ESC 키 누를 시 while 종료
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
print('count = ',count)
print('average time = ',ave_time)
