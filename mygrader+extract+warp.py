#python mygrader.py --image rollnumberonly.jpg
#this program reads the image and draws contours over the bubbles and the numbers

# import the necessary packages
import mygrader
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"
biggestcontour = []

def read_process_image(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 10, 200)
	cv2.imshow("edged", edged)
	cv2.waitKey(0)
	im = edged.copy()
	return edged

def get_draw_contours(im,ims,flag):
	cnts = cv2.findContours(im, cv2.RETR_EXTERNAL,
						   cv2.CHAIN_APPROX_SIMPLE)
	pcsdimage = ims.copy()
	#print('bef imutils grabcontours cnts', len(cnts))
	cnts = imutils.grab_contours(cnts)
	print('after cnts', len(cnts))
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	#biggestcontour = cnts[0]
	#print('biggestcontour in f1', biggestcontour)
	screenCnt = 0
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)
		if len(approx) == 4:
		   screenCnt = approx
		   break
	cv2.drawContours(pcsdimage, [screenCnt], -1, (0, 255, 0), 3)
	cv2.imshow("Outline", pcsdimage)
	cv2.waitKey()
	#ratio = pcsdimage.shape[0] / 200.0
	warped = four_point_transform(pcsdimage, screenCnt.reshape(4, 2))# * ratio)
	cv2.imshow("Scanned", warped)
	cv2.waitKey(0)
	return


def draw_biggest_contour_extract(imb,biggestcontour):
	#imb = image.copy()
	#print('biggestcontour in f2',biggestcontour)
	perib = cv2.arcLength(biggestcontour, True)
	approxb = cv2.approxPolyDP(biggestcontour, 0.3 * perib, True)
	x, y, w, h = cv2.boundingRect(biggestcontour)
	cv2.rectangle(imb, (x, y), ((x + w), (y + h)), (0, 0, 255), 1)
	# cv2.drawContours(imb,biggestcontour,-1,(0,0,255),3)
	cv2.imshow("biggestcontour", imb)  # biggest contour is the name head which is cropped and saved to be read later
	cv2.waitKey()
	cropimage = 'cropped' + str(int(y)) + ".jpg"
	cv2.imwrite(cropimage, imb[y:y + h, x:x + w])
	cv2.imshow(imb[y:y + h, x:x + w])
	cv2.imread(cropimage)
	data = pytesseract.image_to_string(cropimage, lang='eng', config='--psm 6')
	#print(cropimage)
	return cropimage

def get_name_from_crop(cropimagename,ims):
	print('Cropimagepath ',cropimagename)
	croppedimage = cv2.imread(cropimagename)
	cv2.imshow('Cropimageoriginal',croppedimage)
	cv2.waitKey()
	gray = cv2.cvtColor(croppedimage, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Cropimagegray',gray)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 10, 200)
	cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	pcsdimage = croppedimage.copy()
	namecontour =cnts[0] #biggest contour is the one with the name
	perin = cv2.arcLength(namecontour, True)
	approxn = cv2.approxPolyDP(namecontour, 0.2 * perin, True)
	x, y, w, h = cv2.boundingRect(approxn)
	im = cv2.rectangle(pcsdimage, (x, y), ((x + w), (y + h)), (0, 255,0), 1)
	cv2.imshow('name exrtact',pcsdimage)
	cv2.waitKey()
	namepart = pcsdimage[y:y + h, x:x + w]
	namepath = 'name' + str(int(y)) + ".jpg"
	#cv2.imwrite(nameextract, namepart)
	cv2.imread(namepath)
	#data = pytesseract.image_to_string(namepath, lang='eng', config='--psm 6')
	data = pytesseract.image_to_string(namepath, lang='eng', config='--psm 6')
	print(data) #name alone is extracted here


	""""for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)
		x, y, w, h = cv2.boundingRect(approx)
		im = cv2.rectangle(pcsdimage,(x,y),((x+w),(y+h)),(0,0,255),1)
		#cv2.imshow("1-by-1 in crop", pcsdimage)
		#cv2.waitKey()
	#cv2.imshow('All crop-rectboxes',pcsdimage)
	#cv2.waitKey()"""
	return
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
					help="path to the input image")
	args = vars(ap.parse_args())
	image = cv2.imread(args["image"])
	ims = image.copy()
	cv2.imshow("Original Image", image)
	cv2.waitKey(0)
	edged = read_process_image(image)
	im = edged.copy()
	biggestcontour = get_draw_contours(im,ims,flag=1) #flag=1 denotes need to warp it
	cropimage = draw_biggest_contour_extract(image,biggestcontour)
	#print(crop.shape)

	#biggestcontouraa = get_draw_contours(crop, ims)
	#get_name_from_crop(cropimage,ims)


if __name__ == "__main__":
	main()









