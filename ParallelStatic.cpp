#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "mpi.h"
#include <math.h>

/*
    Different Tags to Send Message
*/
#define NORMAL_FRAME 4
#define NORMAL_RESULT 5
#define GRAY_RESULT 6
#define BLUE_RESULT 7
#define GREEN_RESULT 8
#define RED_RESULT 9

using namespace cv;

int main(int argc, char** argv )
{
    int size, rank;
    int i;
    

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status status;
    //Dimension variable to communicate dimension of frame
    int *dimension = (int *)malloc(sizeof(int) * 2);
    /* 
        If Rank = 0 Read video from file, break it into frames and send to other nodes for processing.
    */
    if(rank == 0){
       VideoCapture cap(argv[1]); // Open File
        if(!cap.isOpened())  // check if we succeeded
            return -1;

        Mat edges;
        Mat frame;
        //index each frame to send to different nodes
        int index = 0;
        //namedWindow("edges",1);
        for(;;)
        {
            
            cap >> frame; // get a new frame from camera/video
            //make a continuous frame
            if (!frame.isContinuous())
            { 
                frame = frame.clone();
            }
            
            //Send first frame to all nodes
            if(index == 0){
                //frame = imread( "Lena.jpg", 1 );
                dimension[0] = frame.rows;
                dimension[1] = frame.cols;

                unsigned char* sendMatrix = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1] * 3));

                sendMatrix = frame.data;
                //Broadcast Dimension of the frame
                for(i = 1; i < size; i++){
                    MPI_Send(&dimension[0], 2, MPI_INT, i, index, MPI_COMM_WORLD);
                }
                //Broadcast firstframe
                for(i = 1; i < size; i++){
                    MPI_Send(&sendMatrix[0], dimension[0] * dimension[1] * 3, MPI_UNSIGNED_CHAR, i, index, MPI_COMM_WORLD);
                }
            }
            /*
            Start sending frames but dont send to 
            Size - 1 = Normal Image Result
            Size - 2 = GrayScale Image Result
            Size - 3 = Blue Image Result
            Size - 4 = Green Image Result
            Size - 5 = Red Image Result
            */
            if(index > 0 && index < (size - 5)){
                //Start sending other frames
                unsigned char* sendMatrix = (unsigned char*)malloc(sizeof(unsigned char) * frame.rows * frame.cols * 3);
                sendMatrix = frame.data;
                MPI_Send(&sendMatrix[0], frame.rows * frame.cols * 3, MPI_UNSIGNED_CHAR, index, NORMAL_FRAME, MPI_COMM_WORLD);
                //printf("Sent to %d\n", index);
            }
            //Else Reset index
            else if(index >= (size - 5)){
                index = 0;
            }
            index++;
            if(waitKey(30) >= 0) break;
        }
    }
    else {
        // Receive Initial Background frame
        Mat  backgroundRed, backgroundGreen, backgroundBlue, backgroundGray;
        //Recieve Dimension of the Frame
        MPI_Recv(&dimension[0], 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        unsigned char* recvMat = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1] * 3));
        //Recieve first frame
        MPI_Recv(&recvMat[0], dimension[0] * dimension[1] * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        Mat backgroundImage(dimension[0],dimension[1], CV_8UC3, recvMat);
        //Convert to Grayscale
        cvtColor( backgroundImage, backgroundGray, CV_BGR2GRAY );
        //Split into RGB
        Mat channelBackground[3];
        split (backgroundImage, channelBackground); // channelBackground[0] = blue, 1 = green, 2 = red

        //start Receiving frames       
        if(rank < size - 5){
            Mat temp1, temp1Gray; 
            Mat temp2, temp2Gray; 
            Mat channel1[3];
            Mat channel2[3];
            int j, k, l;
            unsigned char *recvMatrix;
            Vec3b pixel1, pixel1Gray, pixel1Blue, pixel1Green, pixel1Red;
            Vec3b pixel2, pixel2Gray, pixel2Blue, pixel2Green, pixel2Red;
            Vec3b pixelRef, pixelRefGray, pixelRefBlue, pixelRefGreen, pixelRefRed;
            int counter = 0; // To ignore first receive
            long int threshold = 100;
            long int threshold2 = 50;
            long int thresholdGray = 80;
            long int thresholdGray2 = 40;
            long int thresholdBlue = 80;
            long int thresholdBlue2 = 40;
            long int thresholdGreen = 80;
            long int thresholdGreen2 = 40;
            long int thresholdRed = 80;
            long int thresholdRed2 = 40;
            long int ref, val1, val2;

            // Receive frame from master rank = 0
            for(;;){
                recvMatrix = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1] * 3));
                MPI_Recv(&recvMatrix[0], dimension[0] * dimension[1] * 3, MPI_UNSIGNED_CHAR, 0, NORMAL_FRAME, MPI_COMM_WORLD, &status);
                Mat newImage(dimension[0],dimension[1], CV_8UC3, recvMatrix);
                if(counter > 0){
                    
                    temp2 = newImage.clone();
                    //ConvertToGrayScale
                    cvtColor( temp2, temp2Gray, CV_BGR2GRAY );
                    
                    //Split into BGR
                    split (temp2, channel2);
                    if (!temp2Gray.isContinuous())
                    { 
                        temp2Gray = temp2Gray.clone();
                    }
                    //Start checking the difference
                    for(j = 0; j < temp2.cols; j++){
                        for(k = 0; k < temp2.rows; k++){
                            if(j == 0 && k == 0){
                                pixelRef = backgroundImage.at<Vec3b>(Point(j, k));
                                pixelRefGray = backgroundGray.at<uchar>(Point(j, k));
                                pixelRefBlue = channelBackground[0].at<uchar>(Point(j, k));
                                pixelRefGreen = channelBackground[1].at<uchar>(Point(j, k));
                                pixelRefRed = channelBackground[2].at<uchar>(Point(j, k));
                            }

                            pixel1 = temp1.at<Vec3b>(Point(j, k));
                            pixel2 = temp2.at<Vec3b>(Point(j, k));

                            pixel1Gray = temp1Gray.at<uchar>(Point(j, k)); 
                            pixel2Gray = temp2Gray.at<uchar>(Point(j, k));
                            
                            pixel1Blue = channel1[0].at<uchar>(Point(j, k)); 
                            pixel2Blue = channel2[0].at<uchar>(Point(j, k));

                            pixel1Green = channel1[1].at<uchar>(Point(j, k)); 
                            pixel2Green = channel2[1].at<uchar>(Point(j, k));

                            pixel1Red = channel1[2].at<uchar>(Point(j, k)); 
                            pixel2Red = channel2[2].at<uchar>(Point(j, k));

                            //Check difference for normal image
                            ref = (pixelRef[0] + pixelRef[1] + pixelRef[2]);
                            val1 = (pixel1[0] + pixel1[1] + pixel1[2]);
                            val2 = (pixel2[0] + pixel2[1] + pixel2[2]);
                            if(abs(ref - val1) > threshold && abs(val1 - val2) < threshold2){
                                //New static pixel detected
                                temp1.at<Vec3b>(Point(j,k)) = 0;
                            }
        
                            //Check difference for grayscale image
                            ref = (pixelRefGray[0]);
                            val1 = (pixel1Gray[0]);
                            val2 = (pixel2Gray[0]);
                            if(abs(ref - val1) > thresholdGray && abs(val1 - val2) < thresholdGray2){
                                //New static pixel detected
                                temp1Gray.at<uchar>(Point(j,k)) = 255;
                            }

                            //Check difference for blue image
                            ref = pixelRefBlue[0];
                            val1 = (pixel1Blue[0]);
                            val2 = (pixel2Blue[0]);
                            //printf("%d: Ref - val = %ld Val1 - Val2 = %ld\n", rank, abs(ref - val1), abs(val1 - val2));
                            if(abs(ref - val1) > thresholdBlue && abs(val1 - val2) < thresholdBlue2){
                                //New static pixel detected
                                channel1[0].at<uchar>(Point(j,k)) = 255;
                            }

                            //Check difference for Green image
                            ref = (pixelRefGreen[0]);
                            val1 = (pixel1Green[0]);
                            val2 = (pixel2Green[0]);
                            if(abs(ref - val1) > thresholdGreen && abs(val1 - val2) < thresholdGreen2){
                                //New static pixel detected
                                channel1[1].at<uchar>(Point(j,k)) = 255;
                            }

                            //Check difference for Red image
                            ref = (pixelRefRed[0]);
                            val1 = (pixel1Red[0]);
                            val2 = (pixel2Red[0]);
                            if(abs(ref - val1) > thresholdRed && abs(val1 - val2) < thresholdRed2){
                                //New static pixel detected
                                channel1[2].at<uchar>(Point(j,k)) = 255;
                            }
                        }
                    }
                    //Send Normal Image Frame to destination
                    unsigned char* sendMatrix = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1] * 3));
                    sendMatrix = temp1.data;
                    MPI_Send(&sendMatrix[0], dimension[0] * dimension[1] * 3, MPI_UNSIGNED_CHAR, (size - 1), NORMAL_RESULT, MPI_COMM_WORLD);

                    //Send GrayScale to size - 2
                    unsigned char* sendMatrixGray = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1]));
                    sendMatrixGray = temp1Gray.data;
                    MPI_Send(&sendMatrixGray[0], dimension[0] * dimension[1], MPI_UNSIGNED_CHAR, (size - 2), GRAY_RESULT, MPI_COMM_WORLD);
                    

                    //Send Blue to size - 3
                    unsigned char* sendMatrixBlue = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1]));
                    sendMatrixBlue = channel1[0].data;
                    MPI_Send(&sendMatrixBlue[0], dimension[0] * dimension[1], MPI_UNSIGNED_CHAR, (size - 3), BLUE_RESULT, MPI_COMM_WORLD);


                    //Send Green to size - 4
                    unsigned char* sendMatrixGreen = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1]));
                    sendMatrixGreen = channel1[1].data;
                    MPI_Send(&sendMatrixGreen[0], dimension[0] * dimension[1], MPI_UNSIGNED_CHAR, (size - 4), GREEN_RESULT, MPI_COMM_WORLD);
                    

                    //Send Red to size - 5
                    unsigned char* sendMatrixRed = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1]));
                    sendMatrixRed = channel1[2].data;
                    MPI_Send(&sendMatrixRed[0], dimension[0] * dimension[1], MPI_UNSIGNED_CHAR, (size - 5), RED_RESULT, MPI_COMM_WORLD);

                    //Put Second frame data to first Frame to calculate next static objects
                    temp1 = temp2.clone();
                    temp1Gray = temp2Gray.clone();
                    channel1[0] = channel2[0].clone();
                    channel1[1] = channel2[1].clone();
                    channel1[2] = channel2[2].clone();
                }
                else{
                    temp1 = newImage.clone(); 
                    //Convert to GrayScale
                    cvtColor( temp1, temp1Gray, CV_BGR2GRAY );
                    
                    //Split into BGR
                    split (temp1, channel1);
                    if (!temp1Gray.isContinuous())
                    { 
                        temp1Gray = temp1Gray.clone();
                    }
                    counter = 1;
                }  
            }
        }

        //Collect Normal Frame Results
        else if ( rank == (size - 1)){
            int counter = 1;
            VideoWriter video("out_Normal.avi",CV_FOURCC('M','J','P','G'),10, Size(dimension[1], dimension[0]),true);
            for(;;){
                if(counter >= size - 5)
                    counter = 1;
                //Recive Normal Image Result
                unsigned char* resultMat = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1] * 3));
                MPI_Recv(&resultMat[0], dimension[0] * dimension[1] * 3, MPI_UNSIGNED_CHAR, counter, NORMAL_RESULT, MPI_COMM_WORLD, &status);
                Mat resultImage(dimension[0],dimension[1], CV_8UC3, resultMat);
                video.write(resultImage);
                counter++;
            }
        }

        //Collect Grayscale Frame Results
        else if ( rank == (size - 2)){
            int counter = 1;
            VideoWriter video("out_gray.avi",CV_FOURCC('M','J','P','G'),10, Size(dimension[1], dimension[0]), false);
            for(;;){
                if(counter >= size - 5)
                    counter = 1;
                //Recive Gray Image Result
                unsigned char* resultMat = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1]));
                MPI_Recv(&resultMat[0], dimension[0] * dimension[1], MPI_UNSIGNED_CHAR, counter, GRAY_RESULT, MPI_COMM_WORLD, &status);
                Mat resultImage(dimension[0],dimension[1], CV_8UC1, resultMat);
                video.write(resultImage);
                counter++;
            }
        }

        //Collect Blue Channel Frame Results
        else if ( rank == (size - 3)){
            int counter = 1;
            VideoWriter video("out_blue.avi",CV_FOURCC('M','J','P','G'),10, Size(dimension[1], dimension[0]),false);
            for(;;){
                if(counter >= size - 5)
                    counter = 1;
                //Recive Blue Image Result
                unsigned char* resultMat = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1]));
                MPI_Recv(&resultMat[0], dimension[0] * dimension[1], MPI_UNSIGNED_CHAR, counter, BLUE_RESULT, MPI_COMM_WORLD, &status);
                Mat resultImage(dimension[0],dimension[1], CV_8UC1, resultMat);
                video.write(resultImage);
                counter++;
            }
        }

        //Collect Green Channel Frame Results
        else if ( rank == (size - 4)){
            int counter = 1;
            VideoWriter video("out_green.avi",CV_FOURCC('M','J','P','G'),10, Size(dimension[1], dimension[0]),false);
            for(;;){
                if(counter >= size - 5)
                    counter = 1;
                //Recive Green Image Result
                unsigned char* resultMat = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1]));
                MPI_Recv(&resultMat[0], dimension[0] * dimension[1], MPI_UNSIGNED_CHAR, counter, GREEN_RESULT, MPI_COMM_WORLD, &status);
                Mat resultImage(dimension[0],dimension[1], CV_8UC1, resultMat);
                video.write(resultImage);
                counter++;
            }
        }

        //Collect Blue Red Frame Results
        else if ( rank == (size - 5)){
            int counter = 1;
            VideoWriter video("out_red.avi",CV_FOURCC('M','J','P','G'),10, Size(dimension[1], dimension[0]),false);
            for(;;){
                if(counter >= size - 5)
                    counter = 1;
                //Recive Red Image Result
                unsigned char* resultMat = (unsigned char*)malloc(sizeof(unsigned char) * (dimension[0] * dimension[1]));
                MPI_Recv(&resultMat[0], dimension[0] * dimension[1], MPI_UNSIGNED_CHAR, counter, RED_RESULT, MPI_COMM_WORLD, &status);
                Mat resultImage(dimension[0],dimension[1], CV_8UC1, resultMat);
                video.write(resultImage);
                counter++;
            }
        }
    }
    MPI_Finalize();
    return 0;
}