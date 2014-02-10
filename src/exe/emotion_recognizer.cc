#include <Tracker.h>
#include <opencv/highgui.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <svm.h>
#include <string.h>
//=============================================================================
cv::Point cPoints[66];      //an array that stores 66 characteristic points
double featureScaler;       //used to calculate local feature
struct svm_node *node;      //svm node
struct svm_model *svmModel;    //svm model
cv::Point textArea;

double innerBrowRaiser;     //AU 1
double outerBrowRaiser;     //AU 2
double browLower;           //AU 4
double upperLidRaiser;      //AU 5
double lidTightener;        //AU 7
double noseWrinkler;        //AU 9
double lipCornerPull;       //AU 12
double lipCornerDepress;    //AU 15
double lowerLipDepress;     //AU 16
double lipStretch;          //AU 20
double lipTightener;        //AU 23
double jawDrop;             //AU 26

double getDist(cv::Point p1, cv::Point p2)
{
    double r = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    return r;    
}

double getDistX(cv::Point p1, cv::Point p2)
{
    double r = abs(p1.x - p2.x);
    return r;
}

double getDistY(cv::Point p1, cv::Point p2)
{
    double r = abs(p1.y - p2.y);
    return r;
}

void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
  int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;

  //draw triangulation
  c = CV_RGB(0,0,0);
  for(i = 0; i < tri.rows; i++){
    if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
       visi.at<int>(tri.at<int>(i,1),0) == 0 ||
       visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
		   shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
		   shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
		   shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
		   shape.at<double>(tri.at<int>(i,2)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
		   shape.at<double>(tri.at<int>(i,2)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
		   shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
  }
  //draw connections
  c = CV_RGB(0,0,255);
  for(i = 0; i < con.cols; i++){
    if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
       visi.at<int>(con.at<int>(1,i),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
		   shape.at<double>(con.at<int>(0,i)+n,0));
    p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
		   shape.at<double>(con.at<int>(1,i)+n,0));
    cv::line(image,p1,p2,c,1);
  }
    CvFont font;
	cvInitFont(&font,CV_FONT_VECTOR0,1.5,0.8,0.0,1);
	
  //draw points
  for(i = 0; i < n; i++){    
    if(visi.at<int>(i,0) == 0)continue;
    p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
    cPoints[i] = p1;
    //std::cout<<i<<": "<<p1.x<<" "<<p1.y<<std::endl;
    c = CV_RGB(255,0,0);
    cv::circle(image,p1,2,c);
  }return;
}
//=============================================================================
int parse_cmd(int argc, const char** argv,
	      char* ftFile,char* conFile,char* triFile,char* mdFile,
	      bool &fcheck,double &scale,int &fpd)
{
  int i; fcheck = false; scale = 1; fpd = -1;
  for(i = 1; i < argc; i++){
    if((std::strcmp(argv[i],"-?") == 0) ||
       (std::strcmp(argv[i],"--help") == 0)){
      std::cout << "track_face:- Written by Zhanwen Xu 2012" << std::endl
	   << "Performs automatic emotion recognition" << std::endl << std::endl
	   << "#" << std::endl 
	   << "# usage: ./face_tracker [options]" << std::endl
	   << "#" << std::endl << std::endl
	   << "Arguments:" << std::endl
	   << "-m <string> -> Tracker model (default: ../model/face2.tracker)"
	   << std::endl
	   << "-c <string> -> Connectivity (default: ../model/face.con)"
	   << std::endl
     << "-svm <string> -> SVM model (default: ../model/face2.model)"
     << std::endl
	   << "-t <string> -> Triangulation (default: ../model/face.tri)"
	   << std::endl
	   << "-s <double> -> Image scaling (default: 1)" << std::endl
	   << "-d <int>    -> Frames/detections (default: -1)" << std::endl
	   << "--check     -> Check for failure" << std::endl;
      return -1;
    }
  }
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"--check") == 0){fcheck = true; break;}
  }
  if(i >= argc)fcheck = false;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-s") == 0){
      if(argc > i+1)scale = std::atof(argv[i+1]); else scale = 1;
      break;
    }
  }
  if(i >= argc)scale = 1;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-d") == 0){
      if(argc > i+1)fpd = std::atoi(argv[i+1]); else fpd = -1;
      break;
    }
  }
  if(i >= argc)fpd = -1;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-m") == 0){
      if(argc > i+1)std::strcpy(ftFile,argv[i+1]);
      else strcpy(ftFile,"../model/face2.tracker");
      break;
    }
  }
  
  if(i >= argc)std::strcpy(ftFile,"../model/face2.tracker");
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-c") == 0){
      if(argc > i+1)std::strcpy(conFile,argv[i+1]);
      else strcpy(conFile,"../model/face.con");
      break;
    }
  }
  if(i >= argc)std::strcpy(conFile,"../model/face.con");
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-t") == 0){
      if(argc > i+1)std::strcpy(triFile,argv[i+1]);
      else strcpy(triFile,"../model/face.tri");
      break;
    }
  }
  if(i >= argc)std::strcpy(triFile,"../model/face.tri");
  
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-svm") == 0){
      if(argc > i+1)std::strcpy(mdFile,argv[i+1]);
      else strcpy(mdFile,"../model/face2.model");
      break;
    }
  }
  if(i >= argc)std::strcpy(mdFile,"../model/face2.model");
  return 0;
}
//=============================================================================
int main(int argc, const char** argv)
{
  
  //parse command line arguments
  char ftFile[256],conFile[256],triFile[256],mdFile[256];
  bool fcheck = false; double scale = 1; int fpd = -1; bool show = true;
  if(parse_cmd(argc,argv,ftFile,conFile,triFile,mdFile,fcheck,scale,fpd)<0)return 0;

  //set other tracking parameters
  std::vector<int> wSize1(1); wSize1[0] = 7;
  std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
  int nIter = 5; double clamp=3,fTol=0.01; 
  FACETRACKER::Tracker model(ftFile);
  cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
  cv::Mat con=FACETRACKER::IO::LoadCon(conFile);
  
  //initialize camera and display window
  cv::Mat frame,gray,im; double fps=0; char sss[256]; std::string text; 
  CvCapture* camera = cvCreateCameraCapture(CV_CAP_ANY); if(!camera)return -1;
  int64 t1,t0 = cvGetTickCount(); int fnum=0;
  cvNamedWindow("Emotion Recognizer",1);
  std::cout << "Hot keys: "        << std::endl
	    << "\t ESC - quit"     << std::endl
	    << "\t d   - Redetect" << std::endl;

  //load svm model
  if ((svmModel = svm_load_model(mdFile))==0) {
    printf("can not open model file.\n");
    exit(0);
    
  }
  //allocate memory for svm node
  node = (struct svm_node *) malloc(64 * sizeof(struct svm_node));
  
  //loop until quit (i.e user presses ESC)
  bool failed = true;
  while(1){ 
    //grab image, resize and flip
    IplImage* I = cvQueryFrame(camera); 
    //IplImage* I = cvLoadImage("/Users/leonardo/Documents/FACS Cohn Kanade Database/cohn-kanade/S010/001/S010_001_01594215.png");
    if(!I)continue; 
    frame = I;
    if(scale == 1)im = frame; 
    else cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));
    cv::flip(im,im,1); cv::cvtColor(im,gray,CV_BGR2GRAY);

    //track this image
    std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1; 
    if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
      int idx = model._clm.GetViewIdx(); failed = false;
      Draw(im,model._shape,con,tri,model._clm._visi[idx]);
      //assign feature scaler as the width of the face, which does not change in response to different expression
      featureScaler = (getDistX(cPoints[0], cPoints[16]) + getDistX(cPoints[1], cPoints[15]) + getDistX(cPoints[2], cPoints[14]))/3;
      //assign action unit 1
      innerBrowRaiser = ((getDistY(cPoints[21], cPoints[27]) + getDistY(cPoints[22], cPoints[27]))/2) / featureScaler;
      //assign action unit 2
      outerBrowRaiser = ((getDistY(cPoints[17], cPoints[27]) + getDistY(cPoints[26], cPoints[27]))/2) / featureScaler;
      //assign action unit 4
      browLower = (((getDistY(cPoints[17], cPoints[27])+getDistY(cPoints[18], cPoints[27])+
                  getDistY(cPoints[19], cPoints[27])+getDistY(cPoints[20], cPoints[27])+
                  getDistY(cPoints[21], cPoints[27]))/5 + 
                  (getDistY(cPoints[22], cPoints[27])+getDistY(cPoints[23], cPoints[27])+
                  getDistY(cPoints[24], cPoints[27])+getDistY(cPoints[25], cPoints[27])+
                  getDistY(cPoints[26], cPoints[27]))/5) / 2) / featureScaler;
      //assign action unit 5
      upperLidRaiser = ((getDistY(cPoints[37], cPoints[27]) + getDistY(cPoints[44], cPoints[27]))/2) / featureScaler;
      //assign action unit 7
      lidTightener = ((getDistY(cPoints[37], cPoints[41]) + getDistY(cPoints[38], cPoints[40])) / 2 + 
                        (getDistY(cPoints[43], cPoints[47]) + getDistY(cPoints[44], cPoints[46])) / 2) / featureScaler;
      //assign action unit 9
      noseWrinkler = (getDistY(cPoints[29], cPoints[27]) + getDistY(cPoints[30], cPoints[27])) / featureScaler;
      //assign action unit 12
      lipCornerPull = ((getDistY(cPoints[48], cPoints[33]) + getDistY(cPoints[54], cPoints[33])) / 2) / featureScaler;
      //assign action unit 16
      lowerLipDepress = getDistY(cPoints[57], cPoints[33]) / featureScaler;
      //assign action unit 20
      lipStretch = getDistX(cPoints[48], cPoints[54]) / featureScaler;
      //assign action unit 23
      lipTightener = (getDistY(cPoints[49], cPoints[59]) + 
                    getDistY(cPoints[50], cPoints[58]) + 
                    getDistY(cPoints[51], cPoints[57]) + 
                    getDistY(cPoints[52], cPoints[56]) + 
                      getDistY(cPoints[53], cPoints[55]))/featureScaler;
      //assign action unit 26
      jawDrop = getDistY(cPoints[8], cPoints[27]) / featureScaler;      
      
      double class_nr = 0;
      int class_nr_int = 0;
      int i = 0;
      std::string notification;
      
      //predict
      for (i = 0; i < 11; i ++) {
        node[i].index = i;
      }
      node[11].index = -1;
      
      //assign value of nodes
      node[0].value = innerBrowRaiser;
      node[1].value = outerBrowRaiser;
      node[2].value = browLower;
      node[3].value = upperLidRaiser;
      node[4].value = lidTightener;
      node[5].value = noseWrinkler;
      node[6].value = lipCornerPull;
      node[7].value = lowerLipDepress;
      node[8].value = lipStretch;
      node[9].value = lipTightener;
      node[10].value = jawDrop;
      
      //predict the class
      //0: neutral face
      //1: happy
      //2: angry
      //3: disgust
      //-1 sad
      //-2 suprise
      //-3 fear
      class_nr = svm_predict(svmModel, node);
      class_nr_int = (int)class_nr;
      textArea = cv::Point(cPoints[8].x - 50, cPoints[8].y +50);
      if (class_nr_int == 2) {
        notification = "Angry";
      }
      else if (class_nr_int == 1){
        notification = "Happy";
      }
      else if (class_nr_int == 3){
        notification = "Disgust";
      }
      else if (class_nr_int == -1){
        notification = "Sad";
      }
      else if (class_nr_int == -2){
        notification = "Surprise";
      }
      else if (class_nr_int == -3){
        notification = "Fear";
      }
      else
      {
        notification = "Neutral";
      }
      
      CvFont font;
      cvInitFont(&font,CV_FONT_VECTOR0,1.5,0.8,0.0,1);
      cvPutText(I,notification.c_str(),textArea,&font,cvScalar(255));
      

      /*std::cout<<"-3 "<<"0:"<<innerBrowRaiser<<" 1:"<<outerBrowRaiser<<" 2:"<<browLower<<" 3:"<<upperLidRaiser<<" 4:"<<lidTightener<<" 5:"<<noseWrinkler
      <<" 6:"<<lipCornerPull<<" 7:"<<lowerLipDepress<<" 8:"<<lipStretch<<" 9:"<<lipTightener<<" 10:"<<jawDrop<<std::endl;*/
    }else{
      if(show){cv::Mat R(im,cvRect(0,0,150,50)); R = cv::Scalar(0,0,255);}
      model.FrameReset(); failed = true;
    }     
    //draw framerate on display image 
    if(fnum >= 9){      
      t1 = cvGetTickCount();
      fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6); 
      t0 = t1; fnum = 0;
    }else fnum += 1;
    if(show){
      sprintf(sss,"%d frames/sec",(int)round(fps)); text = sss;
      cv::putText(im,text,cv::Point(10,20),
		  CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
    }
    //show image and check for user input
    imshow("Emotion Recognizer",im); 
    int c = cvWaitKey(10);
    if(c == 27)
    {
      svm_free_and_destroy_model(&svmModel);
      free(node);
      break;
    } 
    else if(char(c) == 'd')model.FrameReset();
  }
  
  return 0;
}