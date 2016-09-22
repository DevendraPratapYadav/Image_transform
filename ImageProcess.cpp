#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iomanip>


#define im(x,y)  img.at<uchar>(x,y)
#define imo(x,y)  nimg.at<uchar>(x,y)
#define C(x,y)  C.at<uchar>(x,y)

#define scu(x) saturate_cast<uchar> (x)

#define PI 3.14159265


#include <sstream>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}


using namespace patch;
using namespace cv;
using namespace std;


int getSize(int r,int c,int t,double x,double y)
{


if (t==0)
{
 return max(r+y,c+x);
}

if (t==1)
{
 return max( r*y,  c*x );
}

if (t==2)
{
 return sqrt( r*r+ c*c )+1;
}
if (t==3)
{
 return max(r+c*y,c+x*r);
}

return max(r,c);

} // end getsize


//Mat iRotate(Mat imgo, double tx,double ty, double sx, double sy, double rot, double skx, double sky)

vector< pair<int,int> > getNbr(int i,int j,int H, int W)
{
  vector< pair<int,int> > nb;
  cout<<H<<W<<endl;
  nb.push_back(make_pair(i-1,j));
  nb.push_back(make_pair(i,j+1));
  nb.push_back(make_pair(i+1,j));
  nb.push_back(make_pair(i,j-1));

  for (int i=0;i<nb.size();i++)
	{

	 int y=nb[i].first,x=nb[i].second;
 	   //cout<<it->second<<" "<<it->first<<endl;
	  if ( (x<0 || x>=W) || (y<0 || y>=H))
		{
            cout<<y<<", "<<x<<"||"<<i<<" = "<<(i+2)%4<<endl;
            nb[i]=nb[ (i+2)%4 ];

        }
        else
        cout<<y<<", "<<x<<endl;

	}

	return nb;

} // end getNbr



void print(vector <pair<int,int> > &a )
{
 for (int i=0;i<a.size();i++)
    cout<<a[i].first<<", "<<a[i].second<<" | ";

 cout<<endl;
}


//int BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y)

int BilinearInterpolation(double yy, double xx,Mat img)  // row , column
{


  int x1=(int)xx; int x2=x1+1;
  int y1=(int) yy; int y2=y1+1;


  vector< pair<int,int> > nb;

  nb.push_back(make_pair(x1,y1));
  nb.push_back(make_pair(x2,y1));
  nb.push_back(make_pair(x2,y2));
  nb.push_back(make_pair(x1,y2));

  double sum=0;
  int f=0;
  for (int i=0;i<nb.size();i++)
	{

	 int x=nb[i].first,y=nb[i].second;
 	   //cout<<it->second<<" "<<it->first<<endl;
	  if ( (x<0 || x>=img.cols) || (y<0 || y>=img.rows))
		{


        }
        else
        {  double v=0;
            v=  (1-abs(xx-x) )* (1- abs(yy-y))  ;


           sum+=v*im(y,x) ;
        }

	}



    return ( int )sum;

}


Mat fixImage(Mat img)
{


 //cout<<img.rows<<", "<<img.cols<<endl;
 int t,b,l,r;
 t=b=l=r=0;

 int i=0;
 for (i=0;i<img.rows;i++)
 {
   int flag=0;
   for (int j=0;j<img.cols;j++)
   {
     if (im(i,j)!=0)
        {flag=1; break;}

   }

   if (flag==1)
        break;

 }
 t=i;

 for (i=img.rows-1;i>=0;i--)
 {
   int flag=0;
   for (int j=0;j<img.cols;j++)
   {
     if (im(i,j)!=0)
        {flag=1; break;}

   }

   if (flag==1)
        break;

 }
 b=i;

int j=0;

for (j=0;j<img.cols;j++)
 {
   int flag=0;
   for (int i=0;i<img.rows;i++)
   {
     if (im(i,j)!=0)
        {flag=1; break;}

   }

   if (flag==1)
        break;

 }
 l=j;

for (j=img.cols-1;j>=0;j--)
 {
   int flag=0;
   for (int i=0;i<img.rows;i++)
   {
     if (im(i,j)!=0)
        {flag=1; break;}

   }

   if (flag==1)
        break;

 }
 r=j;

Mat fix(b-t+2,r-l+2,CV_8U,Scalar(0,0,0));

for (int i=0;i<fix.rows;i++)
 {

   for (int j=0;j<fix.cols;j++)
   {
        fix.at<uchar>(i,j)=img.at<uchar>(i+t,j+l);

   }



 }

//printf("%d %d %d %d\n",t,b,l,r);

return fix;


} // end fixImage



Mat iRotate(Mat imgo, double rot,int bilinear)
{

 if (rot<0)
   rot+=360;

 int row1=0,col1=0,row2=0,col2=0;
 rot=rot*PI/180;

 double aff[3][3];
 aff[0][0]=cos(rot);
 aff[0][1]=-sin(rot);
 aff[0][2]=0;

 aff[1][0]=sin(rot);
 aff[1][1]=cos(rot);
 aff[1][2]=0;

 aff[2][0]=0;
 aff[2][1]=0;
 aff[2][2]=1;



    int mx=(imgo.cols/2);
    int my=(imgo.rows/2);
  for(int i=0;i<imgo.rows;i++) // y
    {
      for (int j=0;j<imgo.cols;j++) // x
      {

           int x=(j-mx) *aff[0][0]+(i-my)*aff[0][1]+aff[0][2];

           int y=(j-mx) *aff[1][0]+(i-my)*aff[1][1]+aff[1][2];

           row1=min(row1,y); row2=max(row2,y);
           col1=min(col1,x); col2=max(col2,x);

      }
    }

    int dg=imgo.rows;//sqrt(imgo.cols*imgo.cols+imgo.rows*imgo.rows)+5;
    int dgx=imgo.cols;//dg;
    //int dgy=max(dg,imgo.cols);
    //dg=max(dg,imgo.rows);


    //int dg=row2-row1+1;
    //int dgx=col2-col1+1;
    //cout<<"New image size: "<<dg<<", "<<dgy<<endl;
    //dg=dgx;

    //printf("%d ,%d | %d, %d\n",row1,row2,col1,col2);

    //Mat nimg(2*max(imgo.rows,row1),2*max(imgo.cols, col1), CV_8U, Scalar(0,0,0));

    Mat nimg(dg,dgx, CV_8U, Scalar(0,0,0));


     for(int i=0;i<imgo.rows;i++) // y
    {
      for (int j=0;j<imgo.cols;j++) // x
      {
          // nimg.at<uchar>(i,j) =imgo.at<uchar>(i,j);
           nimg.at<uchar>(i+(nimg.rows-imgo.rows)/2,j+(nimg.cols-imgo.cols)/2) =imgo.at<uchar>(i,j);

      }
    }

    int midx=(nimg.cols/2);
    int midy=(nimg.rows/2);

    /* namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", nimg );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window

*/
    Mat img(dg,dgx, CV_8U, Scalar(0,0,0));

    double mdx=(img.cols/2);
    double mdy=(img.rows/2);


    for(int i=0;i<img.rows;i++) // y
    {
      for (int j=0;j<img.cols;j++) // x
      {

           double x=(j-mdx)*aff[0][0]+(i-mdy)*aff[1][0]-(aff[0][2]*aff[0][0]+aff[1][2]*aff[1][0]);

           double y=(j-mdx)*aff[0][1]+(i-mdy)*aff[1][1]+-(aff[0][2]*aff[0][1]+aff[1][2]*aff[1][1]);

           if ( (int)y+midy>=0 && (int)y+midy<nimg.cols && (int)x+midx>=0 && (int)x+midx<nimg.rows  )
              {
               if (bilinear==1)
              im(i,j)=  BilinearInterpolation(y+midy,x+midx,nimg);   //
              else
              im(i,j)=imo((int)y+midx,(int)x+midy);
              }


      }
    }


return img;
} // end affine



Mat iTranslate(Mat imgo, double tx,double ty)
{

 double aff[3][3];
 aff[0][0]=1;
 aff[0][1]=0;
 aff[0][2]=tx;

 aff[1][0]=0;
 aff[1][1]=1;
 aff[1][2]=ty;

 aff[2][0]=0;
 aff[2][1]=0;
 aff[2][2]=1;


    int sx=imgo.cols+tx;
    int sy=imgo.rows+ty;

    sx=imgo.cols;//max(imgo.cols,sx);
    sy=imgo.rows;//max(imgo.rows,sy);

    //Mat nimg(2*max(imgo.rows,row1),2*max(imgo.cols, col1), CV_8U, Scalar(0,0,0));

    // namedWindow( "Display window2", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window2", imgo );                   // Show our image inside it.

   // waitKey(0);                                          // Wait for a keystroke in the window


    Mat img(sy,sx, CV_8U, Scalar(0,0,0));

    int mdx=0;
    int mdy=0;


    for(int i=0;i<img.rows;i++) // y
    {
      for (int j=0;j<img.cols;j++) // x
      {

           double x=(j-mdx)*aff[0][0]+(i-mdy)*aff[1][0]-(aff[0][2]*aff[0][0]+aff[1][2]*aff[1][0]);
            //double x=j-tx;
           double y=(j-mdx)*aff[0][1]+(i-mdy)*aff[1][1]-(aff[0][2]*aff[0][1]+aff[1][2]*aff[1][1]);
            //double y=i-ty;

            if (i==j)
            {// printf("%d, %d : %f %f\n",j,i,x,y);
            }

           if ( y>=0 && y<imgo.rows && x>=0 && x<imgo.cols )
            im(i,j)=imgo.at<uchar>((int)y,(int)x);

      }
    }


return img;
} // end affine




Mat iScale(Mat imgo, double tx,double ty, int bilinear)
{
double aff[3][3];
 aff[0][0]=1/tx;
 aff[0][1]=0;
 aff[0][2]=0;

 aff[1][0]=0;
 aff[1][1]=1/ty;
 aff[1][2]=0;

 aff[2][0]=0;
 aff[2][1]=0;
 aff[2][2]=1;


    int sx=imgo.cols*tx;
    int sy=imgo.rows*ty;


    //Mat nimg(2*max(imgo.rows,row1),2*max(imgo.cols, col1), CV_8U, Scalar(0,0,0));

    // namedWindow( "Display window2", WINDOW_AUTOSIZE );// Create a window for display.
   // imshow( "Display window2", imgo );                   // Show our image inside it.

   // waitKey(0);                                          // Wait for a keystroke in the window


    Mat img(sy,sx, CV_8U, Scalar(0,0,0));

    int mdx=0;
    int mdy=0;


    for(int y=0;y<img.rows;y++) // y
    {
      for (int x=0;x<img.cols;x++) // x
      {

           double xx=(x)*aff[0][0]+(y)*aff[1][0]-(aff[0][2]*aff[0][0]+aff[1][2]*aff[1][0]);
            //double x=j-tx;
           double yy=(x)*aff[0][1]+(y)*aff[1][1]-(aff[0][2]*aff[0][1]+aff[1][2]*aff[1][1]);
            //double y=i-ty;

           if ( yy>=0 && yy<imgo.rows && xx>=0 && xx<imgo.cols )
            {
             if (bilinear==1)
            im(y,x)=BilinearInterpolation(yy,xx,imgo);//imgo.at<uchar>((int)yy,(int)xx);
             else
             im(y,x)=imgo.at<uchar>((int)yy,(int)xx);

            }


      }
    }


return img;
} // end affine




Mat iSkew(Mat imgo, double tx,double ty,int bilinear)
{


double aff[3][3];
 aff[0][0]=1;
 aff[0][1]=ty;
 aff[0][2]=0;

 aff[1][0]=tx;
 aff[1][1]=1;
 aff[1][2]=0;

 aff[2][0]=0;
 aff[2][1]=0;
 aff[2][2]=1;



    int sx=imgo.cols+imgo.rows*tx;
    int sy=imgo.rows+imgo.cols*ty;

    sx=sqrt(sx*sx+sy*sy);
    sy=sx;
    Mat nimg(sy,sx, CV_8U, Scalar(0,0,0));


     for(int i=0;i<imgo.rows;i++) // y
    {
      for (int j=0;j<imgo.cols;j++) // x
      {
          // nimg.at<uchar>(i,j) =imgo.at<uchar>(i,j);
           nimg.at<uchar>(i+(nimg.rows-imgo.rows)/2,j+(nimg.cols-imgo.cols)/2) =imgo.at<uchar>(i,j);

      }
    }
    // namedWindow( "Display window2", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window2", nimg );                   // Show our image inside it.

   // waitKey(0);                                          // Wait for a keystroke in the window


    Mat img(sy,sx, CV_8U, Scalar(0,0,0));

    double mdx=img.cols/2;
    double mdy=img.rows/2;


    for(int y=0;y<img.rows;y++) // y
    {
      for (int x=0;x<img.cols;x++) // x
      {

           double xx=(x-mdx)*aff[0][0]+(y-mdy)*aff[1][0]-(aff[0][2]*aff[0][0]+aff[1][2]*aff[1][0]);
            //double x=j-tx;
           double yy=(x-mdx)*aff[0][1]+(y-mdy)*aff[1][1]-(aff[0][2]*aff[0][1]+aff[1][2]*aff[1][1]);
            //double y=i-ty;

           if ( yy+mdy>=0 && yy+mdy<nimg.rows && xx+mdx>=0 && xx+mdx<nimg.cols )
            {
             if (bilinear==1)
            im(y,x)=BilinearInterpolation(yy+mdy,xx+mdx,nimg);//nimg.at<uchar>((int)yy+mdy,(int)xx+mdx);
            else
            im(y,x)=nimg.at<uchar>((int)yy+mdy,(int)xx+mdx);
            }
      }
    }


return img;
} // end affine


void negative(Mat &img)
{

 for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {

           im(i,j)=scu(255-im(i,j));
      }
    }



} // end negative




void logTrans(Mat &img,int inv)
{

 int lv[256]={0},ilv[256]={0};

 for (int i=0;i<256;i++)
    {lv[i]=scu( 255.0 * log( (double)( 1+ i)  ) /  log( 256.0)    );  ilv[lv[i]]=i; }

 for (int i=1;i<256;i++)
    { if (ilv[i]==0)
            ilv[i]=ilv[i-1];
    }


 for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {
            int nv=scu( 255.0 * log( (double)( 1+im(i,j))  ) /  log( 256.0)    );

            if (inv==0)
            im(i,j)=nv;
            else
            im(i,j)=scu(  ilv[im(i,j)]  );//im(i,j)-abs(nv-im(i,j)));

      }
    }



} // end negative


void gamma(Mat &img,double p)
{

 for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {
            im(i,j)=scu( 255*pow(im(i,j)/255.0,p));

      }
    }



} // end negative


void contrastStretching( Mat &img, vector < pair<double,double> > ps )
{
 // ps contains point slope pairs (slope of function just after that point). eg. if transformation function has slope 0.5 till 50, 2 till 200, and 0.5 till 255
 // then ps contains: { (0,0.5) , (50,2) , (200,0.5), (255,0.5)  }  0 and 255 point pairs must be there

  double nv[256]={0};
 int x=0; double v=0;
 for (int i=0; i<ps.size(); i++)
 {

  x=ps[i].first; double s= ps[i].second;
  if (x==255)
  break;

  while(x<ps[i+1].first)
  {nv[x++]=v;
  v=v+s;
  if (x==255)
  break;
  }

 } // end for


 for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {

           im(i,j)=scu(  (int)nv[ im(i,j) ] );
      }
    }



} // end stretch


vector <double> getEqualHistogram(Mat &img)
{

    vector <double> h (256,0.0);

    for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {
            h[ im(i,j) ]++;
          }
        }

     for (int i=1;i<256;i++)
      {
       h[i]+=h[i-1];
       //cout<<i<<": "<<h[i]<<endl;
      }

     for (int i=0;i<256;i++)
        {
         h[i]/=(img.cols)*(img.rows);
         h[i]*=255;
        }

    return h;

} // end get Hist


void histogramEqualize( Mat &img)
{

    /*double nv[256]={0};

    for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {
        nv[ im(i,j) ]++;
      }
    }

    for (int i=0;i<256;i++)
    {
     nv[i]/=(img.cols)*(img.rows);
     nv[i]*=255;
    }
*/
  vector <double> nv=getEqualHistogram(img);

  for (int i=0;i<256;i++)
  {
   //cout<<i<<": "<<nv[i]<<endl;
  }

     for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {

               im(i,j)=scu(  (int)nv[ im(i,j) ] );
          }
        }



} // end hist equalize


void adaptiveHistogramEqualize( Mat &img)
{

  int div=8;
  double rd=(double)img.rows/div;
  double cd=(double)img.cols/div;


  vector <double> block[div+1][div+1];

  for (int i=0;i<=div;i++)
  {
    for (int j=0;j<=div;j++)
    {

        if (i==div)
        {block[i][j]=block[i-1][j]; continue; }


        if (j==div)
        {block[i][j]=block[i][j-1]; continue; }

         Rect roi(j*cd, i*rd, cd, rd);
        //Create the cv::Mat with the ROI you need, where "image" is the cv::Mat you want to extract the ROI from
         Mat image_roi = img(roi);

         block[i][j]=getEqualHistogram(image_roi);

         for (int ii=0;ii<256;ii++)
          {
          // cout<<ii<<": "<<block[i][j][ii]<<endl;
          }

      }
    }

     for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {
              double yy=i/rd, xx=j/cd;

              int x1=(int)xx; int x2=x1+1;
              int y1=(int) yy; int y2=y1+1;



              /*if ( (x1>=div-1) || (y1>=div-1))
              {im(i,j)=scu(  (int)  block[div-1][div-1][im(i,j)] );
               continue;
              }
              */

              vector< pair<int,int> > nb;

              nb.push_back(make_pair(x1,y1));
              nb.push_back(make_pair(x2,y1));
              nb.push_back(make_pair(x2,y2));
              nb.push_back(make_pair(x1,y2));

              double sum=0;
              int f=0;
              for (int k=0;k<nb.size();k++)
                {

                 int x=nb[k].first,y=nb[k].second;
                   //cout<<it->second<<" "<<it->first<<endl;
                  if ( (x<0 || x>div) || (y<0 || y>div))
                    {
                        f++;

                    }
                    else
                    {  double v=0;
                        v=  (1-abs(xx-x) )* (1- abs(yy-y))  ;

                       sum+=v* block[y][x][ im(i,j) ];
                    }

                }

               //vector <double> nv=block[r][c];
               im(i,j)=scu(  (int)  sum );

               /*if (i==j && i%10==0)
               {
                 //cout<<i<<","<<j<<": "<<f<<" | "<<sum<<endl;
               }*/

          }
        }



} // end hist equalize



void histogramMatching(Mat &img, Mat & nimg)
{

 vector <double> oh=getEqualHistogram(img);
 vector <double> nh=getEqualHistogram(nimg);


  for (int i=0;i<256;i++)
  {

    int mindex=i;
    double mini=255;

     for (int j=0;j<256;j++)
     {
          if (  abs(oh[i]-nh[j] )<mini)
          { mini=abs(oh[i]-nh[j]);
           mindex=j; }
     }

     oh[i]=mindex;


   //cout<<i<<": "<<nv[i]<<endl;
  }

     for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {
               im(i,j)=scu(  (int)oh[ im(i,j) ] );
          }
        }

    for (int i=0;i<256;i++)
  {
    //cout<<oh[i]<<" : "<<nh[i]<<endl;
  }



} // end hist matching


Mat spatFilter(double filter[3][3], Mat &img,int sob)
{
  int d=3;  // size of square filter matrix
  Mat nimg(img.rows,img.cols, CV_8U, Scalar(0,0,0));

  for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {
               double val=0;

               for (int r=0;r<d;r++)
               {
                    for (int c=0;c<d;c++)
                    {
                        int rr=i+r-d/2; int cc=j+c-d/2;

                        if (rr<0) rr=0;
                        if (cc<0) cc=0;
                        if (rr>=img.rows) rr=img.rows-1;
                        if (cc>=img.cols) cc=img.cols-1;

                        val+=filter[r][c]*(double)im(rr,cc);
                       // if (i==j && i==20 )
                         //   printf("%d %d: %f + %f | ",i,j,val,filter[r][c]*(double)im(rr,cc));


                    }
               }

                if (sob==1)
                    imo(i,j)=scu(abs(val));
                else
                    imo(i,j)=scu((val));


          }
        }

    return nimg;
}       // end spatFilter


void meanFilter(Mat &img)
{

 double filter[3][3]={ {1,1,1},{1,1,1},{1,1,1} };

int d=3;
for (int r=0;r<d;r++)
   {
        for (int c=0;c<d;c++)
            filter[r][c]/=9;
   }

  img=spatFilter(filter,img,0);


} // end mean



void medianFilter(Mat &img)
{
  int d=3;
  Mat nimg(img.rows,img.cols, CV_8U, Scalar(0,0,0));

  for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {
               vector <int> points;

                for (int r=0;r<d;r++)
               {
                    for (int c=0;c<d;c++)
                    {
                        int rr=i+r-d/2; int cc=j+c-d/2;

                        if (rr<0) rr=0;
                        if (cc<0) cc=0;
                        if (rr>=img.rows) rr=img.rows-1;
                        if (cc>=img.cols) cc=img.cols-1;

                        points.push_back(im(rr,cc));


                    }
               }

               sort(points.begin(), points.end());

               imo(i,j)=points[d*d/2];

          }

        }


   img=nimg;


} // end mean



void weightedMeanFilter(Mat &img)
{

 double filter[3][3]={ {1,2,1},{2,4,2},{1,2,1} };

int d=3;
for (int r=0;r<d;r++)
   {
        for (int c=0;c<d;c++)
            filter[r][c]/=16;
   }

  img=spatFilter(filter,img,0);


} // end mean


void laplacianFilter(Mat &img)
{

    //double filter[3][3]={ {-1,-1,-1},{-1,9,-1},{-1,-1,-1} };
    double filter[3][3]={ {0,-1,0},{-1,5,-1},{0,-1,0} };

      img=spatFilter(filter,img,0);


} // end mean

void laplacianFilter2(Mat &img)
{

    double filter[3][3]={ {-1,-1,-1},{-1,9,-1},{-1,-1,-1} };

      img=spatFilter(filter,img,0);


} // end mean



void sobelFilter(Mat &img)
{

    //meanFilter(img);
    double filter[3][3]={ {-1,-2,-1},{0,0,0},{1,2,1} };
    double filter2[3][3]={ {-1,0,1},{-2,0,2},{-1,0,1} };

      Mat img1=spatFilter(filter,img,1);
        Mat img2=spatFilter(filter2,img,1);
      //add(img1,img2,img);
        for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {

            double Gx=img1.at<uchar>(i,j);
            double Gy=img2.at<uchar>(i,j);

            im(i,j)=scu( (int) sqrt( Gx*Gx+Gy*Gy)/1.3);



           }
        }


} // end mean


void unsharpFilter(Mat &img,double k)
{

      Mat img1(img.rows,img.cols, CV_8U, Scalar(0,0,0));

      for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {
           img1.at<uchar>(i,j)=im(i,j);
          }
        }

        meanFilter(img1);


        for(int i=0;i<img.rows;i++)
        {
          for (int j=0;j<img.cols;j++)
          {
           im(i,j)=scu( im(i,j) + k* (im(i,j)- img1.at<uchar>(i,j) ) );
          }
        }


} // end unsharp





vector < pair<double,double> >  getStretchFunction(double p1,double p1y, double p2,double p2y)
{

    vector < pair<double,double> > ps;

    double s1=(p1y/p1);

    double s3=(255-p2y)/(255-p2) ;

    double s2=(255.0 - ( (p1*s1)+(255-p2)*s3) )/( p2-p1 );
    //cout<<s2<<endl;
    ps.push_back ( make_pair( 0,s1 ) );
    ps.push_back ( make_pair( p1,s2 ) );
    ps.push_back ( make_pair( p2,s3 ) );
    ps.push_back ( make_pair( 255,s3 ) );

    return ps;

}



void bitPlane(Mat &img,int bit)
{

 for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {
            if ( ( (uchar)im(i,j)& (1<<bit)) !=0)
                { im(i,j)=255; }

            else
                im(i,j)=scu(0);
      }
    }

} // end bit plane


void graySlice(Mat &img,int low,int high)
{

 for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {
            if ( im(i,j)<=high && im(i,j)>=low)
                { }

            else
                im(i,j)=scu(0);
      }
    }

} // end bit plane


void show(String name, Mat &img, int wait)
{

 namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
 imshow( name, img );
 if (wait>0)
 waitKey(0);

}

Mat oRotate(Mat src, double angle)
{
    Mat dst;
    Point2f pt(src.cols/2., src.rows/2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
    return dst;
}


double rmse(Mat &img,Mat &nimg)
{


 double sum=0;
 for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {
            int diff=im(i,j)-imo(i,j);

            sum+=diff*diff;


      }
    }

    sum/=img.rows*img.cols;

    sum=sqrt (sum);

   cout<<"RMSE : "<<setprecision(3)<<sum<<endl;

} // end bit plane



Mat oTranslate(Mat &img, double offsetx, double offsety){


    cv::Mat imgTranslated(img.size(),img.type(),cv::Scalar::all(0));
    img(cv::Rect(offsetx,offsety,img.cols-offsetx,img.rows-offsety)).copyTo(imgTranslated(cv::Rect(0,0,img.cols-offsetx,img.rows-offsety)));
    return imgTranslated;
}


Mat oScale(Mat &img, double sx, double sy)
{
Size size(img.cols*sx,img.rows*sy);//the dst image size,e.g.100x100
Mat dst;//dst image

resize(img,dst,size);//resize image

return dst;

}


int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: Please provide image name/path as argument." << endl;
     return -1;
    }

    Mat img;
    img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    if(! img.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image\n Please provide image name/path as argument" << std::endl ;
        return -1;
    }
    String fname="";

    int op=0;


    cout<<"Following operations are available :\n"
    "1. Rotate""\n"
    "2. Translate""\n"
    "3. Scale""\n"
    "4. Skew""\n"
    "5. Negative""\n"
    "6. Log Transformation""\n"
    "7. Inverse Log Transformation""\n"
    "8. Gamma Transformation""\n"
    "9. Contrast Stretching""\n"
    "10. Gray Level Slicing""\n"
    "11. Bit Plane Slicing""\n"
    "12. Histogram Equalization""\n"
    "13. Adaptive Histogram Equalization""\n"
    "14. Histogram Matching""\n"
    "15. Mean Filtering""\n"
    "16. Weighted Mean Filtering""\n"
    "17. Median Filtering""\n"
    "18. Sobel Filtering""\n"
    "19. Laplacian Filtering""\n"
    "20. HighBoost Filtering""\n"
    "21. Unsharp Masking""\n"
    "Enter the number corresponding to the desired operation :"
    <<endl;

    cin>>op;

    int i=op;

    fname+=to_string(op);


    Mat res, ores;

    imwrite("orig_"+fname+".tif", img);

    if (i==1)
    {
     // "1. Rotate""\n"

     double ang=0;
     cout<<"Enter rotation angle (positive angle is Clockwise rotation)"<<endl;
     cin>>ang;
     int bb=0;
     cout<<"Select interpolation: \n1. Bilinear Interpolation\n 2. Nearest Neighbor\n";
     cin>>bb;
     res=iRotate(img,-ang,bb);
     //res=fixImage(res);

     show(fname+"_"+to_string(ang)+"_"+"Mine",res,1);


     ores=oRotate(img,ang);

     show(fname+"_"+to_string(ang)+"_"+"OCV",ores,1);

    rmse(res,ores);

    }
    else if (i==2)
    {
    // "2. Translate""\n"

    double tx=0,ty=0;
     cout<<"Enter translation values : Tx Ty"<<endl;
     cin>>tx>>ty;

     res=iTranslate(img,tx,ty);
     //res=fixImage(res);

     show(fname+"_"+to_string(tx)+","+to_string(ty)+"_"+"Mine",res,1);


     //Mat ores=oTranslate(img,tx,ty);

     //show(fname+"_"+to_string(tx)+","+to_string(ty)+"_"+"OCV",ores,1);

    //rmse(res,ores);



    }
    else if (i==3)
    {
    // "3. Scale""\n"

        double sx=0,sy=0;
     cout<<"Enter Scaling values : Sx Sy"<<endl;
     cin>>sx>>sy;

     int bb=0;
     cout<<"Select interpolation: \n1. Bilinear Interpolation\n 2. Nearest Neighbor\n";
     cin>>bb;
     res=iScale(img,sx,sy,bb);
     //res=fixImage(res);

     show(fname+"_"+to_string(sx)+","+to_string(sy)+"_"+"Mine",res,1);

    ores=oScale(img,sx,sy);

    show(fname+"_"+"_"+"OCV",ores,1);

    rmse(res,ores);


    }
    else if (i==4)
    {
    // "4. Skew""\n"
    double sx=0,sy=0;
     cout<<"Enter Skewing values : Sx Sy"<<endl;
     cin>>sx>>sy;

     int bb=0;
     cout<<"Select interpolation: \n1. Bilinear Interpolation\n 2. Nearest Neighbor\n";
     cin>>bb;

     res=iSkew(img,sx,sy,bb);
     res=fixImage(res);

     show(fname+"_"+to_string(sx)+","+to_string(sy)+"_"+"Mine",res,1);

    }
    else if (i==5)
    {
    // "5. Negative""\n"
     negative(img);
     //res=fixImage(res);

     show(fname+"_"+"Mine",img,1);

    }
    else if (i==6)
    {
   //  "6. Log Transformations""\n"
    logTrans(img,0);

    show(fname+"_"+"Mine",img,1);

    }
    else if (i==7)
    {
    // "7. Inverse Log Transformations""\n"
        logTrans(img,1);

        show(fname+"_"+"Mine",img,1);
    }
    else if (i==8)
    {
   //  "8. Gamma Transformation""\n"
     double ang=0;
     cout<<"Enter Gamma value"<<endl;
     cin>>ang;

     gamma(img,ang);
     //res=fixImage(res);

     show(fname+"_"+to_string(ang)+"_"+"Mine",img,1);

    }
    else if (i==9)
    {
    // "9. Contrast Stretching""\n"

    double x1=0,y1=0,x2,y2;
     cout<<"Enter Points of stretching function:"<<endl;
     cout<<" x1,y1 :";
     cin>>x1>>y1;
     cout<<" x2,y2 :";
     cin>>x2>>y2;

     vector < pair<double,double> > ps;

     ps=getStretchFunction(x1,y1,x2,y2);
     contrastStretching(img,ps);
     //res=fixImage(res);

     show(fname+"_"+"_"+"Mine",img,1);


    }
    else if (i==10)
    {
    // "10. Gray Level Slicing""\n"


    int x1=0,y1=0;
     cout<<"Enter Low and High Threshold values :"<<endl;
     cout<<" Low High:";
     cin>>x1>>y1;
    graySlice(img,x1,y1);

     show(fname+"_"+"_"+"Mine",img,1);

    }
    else if (i==11)
    {
   //  "11. Bit Plane Slicing""\n"

     int x1=0;
     cout<<"Enter bit value to slice (0 to 7) :"<<endl;

     cin>>x1;
     bitPlane(img,x1);

     show(fname+"_"+"_"+"Mine",img,1);

    }
    else if (i==12)
    {
    // "12. Histogram Equalization""\n"
      res=fixImage(img);
      histogramEqualize(res);

     show(fname+"_"+"_"+"Mine",res,1);

      ores=fixImage(img);
      equalizeHist( img, ores );
      show(fname+"_"+"_"+"OCV",ores,1);


        rmse(res,ores);

    }
    else if (i==13)
    {
    // "13. Adaptive Histogram Equalization""\n"

      res=fixImage(img);
      adaptiveHistogramEqualize(res);

      show(fname+"_"+"_"+"Mine",res,1);



        Ptr<CLAHE> clahe = createCLAHE();
        clahe->setClipLimit(6);

        //ores;
        clahe->apply(img,ores);
        show(fname+"_"+"_"+"OCV",ores,1);
        rmse(res,ores);

    }

    else if (i==14)
    {
    // "14. Histogram Matching""\n"
    cout<<"Enter target image path/name : "<<endl;
    string s;
    cin>>s;
    Mat nres=imread(s, CV_LOAD_IMAGE_GRAYSCALE);

     res=fixImage(img);
      histogramMatching(res,nres);

     show(fname+"_"+"_"+"Mine",res,1);


    }
    else if (i==15)
    {
    // "15. Mean Filtering""\n"
        res=fixImage(img);
     meanFilter(res);

      show(fname+"_"+"_"+"Mine",res,1);

        //ores;
        blur( img, ores, Size( 3, 3 ), Point(-1,-1) );

        show(fname+"_"+"_"+"OCV",ores,1);
        rmse(res,ores);
    }
    else if (i==16)
    {
   //  "16. Weighted Mean Filtering""\n"
        res=fixImage(img);
        weightedMeanFilter(res);

        show(fname+"_"+"_"+"Mine",res,1);
        //ores;
        GaussianBlur( img, ores, Size( 3, 3 ), 0, 0 );

        show(fname+"_"+"_"+"OCV",ores,1);
        rmse(res,ores);
    }
    else if (i==17)
    {
   //  "17. Median Filtering""\n"

        res=fixImage(img);
        medianFilter(res);

        show(fname+"_"+"_"+"Mine",res,1);
        //ores;
         medianBlur ( img, ores, 3 );

        show(fname+"_"+"_"+"OCV",ores,1);
        rmse(res,ores);


    }
    else if (i==18)
    {
   //  "18. Sobel Filtering""\n"
        res=fixImage(img);

         GaussianBlur( res, res, Size(3,3), 0, 0, BORDER_DEFAULT );

        sobelFilter(res);

        show(fname+"_"+"_"+"Mine",res,1);

         GaussianBlur( img, img, Size(3,3), 0, 0, BORDER_DEFAULT );
        Mat src_gray=img;
        Mat grad;

        int scale = 1;
        int delta = 0;
        int ddepth = CV_8U;

        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        /// Gradient X

        Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );

        /// Gradient Y
        Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );

        addWeighted( abs_grad_x, 0.8, abs_grad_y, 0.8, 0, grad );


        ores=grad;

        show(fname+"_"+"_"+"OCV",ores,1);
        rmse(res,ores);

    }
    else if (i==19)
    {
    // "19. Laplacian Filtering""\n"

    int n;
    cout <<"Enter no. of neighbors to consider : (4/8) \n";
    cin>>n;

    if (n==4)
    {
      laplacianFilter(img);
    }
    else if (n==8)
    {
      laplacianFilter2(img);
    }
    else
    return 0;

    show(fname+"_"+"_"+"Mine",img,1);


    }
    else if (i==20)
    {
   //  "20. HighBoost Filtering""\n"
      double x1=0;
     cout<<"Enter k value :"<<endl;

     cin>>x1;
    unsharpFilter(img,x1);

     show(fname+"_"+"_"+"Mine",img,1);


    }
    else if (i==21)
    {
    // "21. Unsharp Masking""\n"
      unsharpFilter(img,1);

     show(fname+"_"+"_"+"Mine",img,1);

    }

    else
    {cout<<"Please enter a correct integer\n";}




        fname+=".tif";
    imwrite("mineimg_"+fname, img);
    if (res.cols>1)
    imwrite("mine_"+fname, res);
    if (ores.cols>1)
    imwrite("ocv_"+fname, ores);










   // Mat img2=iRotate(img,20); //img2=iRotate(img2,350);


     //Mat img2=iTranslate(img,-100,-184);
    //Mat img2=iScale(img,1,1);
 //Mat img2=iSkew(img,0,1);
 //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
   // imshow( "Display window", img2 );


    //img2=fixImage(img2);
    //gamma(img2,0.2);


     //vector < pair<double,double> > ps;
     //ps.push_back(make_pair(0,0.5)); ps.push_back(make_pair(50,1.32258) ); ps.push_back(make_pair(205,0.5) ); ps.push_back(make_pair(255,0.5) );

     //ps=getStretchFunction(50,200,0.2,0.3);
     //contrastStretching(img2,ps);


  /*

     Mat img4 = imread("h1.jpg", CV_LOAD_IMAGE_GRAYSCALE);

   //histogramEqualize(img);

    Mat img2=fixImage(img);
     //laplacianFilter(img2);



     namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", img2 );                   // Show our image inside it.

    waitKey(0);
    for (int b=0;b<0;b++)
    {
    img2=fixImage(img);
    //bitPlane(img2,b);
    graySlice(img2,0,b*30);

    namedWindow( "Display window3"+b, WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window3"+b, img2 );
    waitKey(0);
    }


    //laplacianFilter2(img);
    meanFilter(img2);                                   // Wait for a keystroke in the window
    //histogramMatching(img,img4);
    //Mat img3=iScale(img,0.4,0.5);
    //adaptiveHistogramEqualize( img3 );


    namedWindow( "Display window3", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window3", img2 );

    waitKey(0);
    */

   /*vector < pair<int,int> > out;
    print( out=getNbr(0,0,img.rows,img.cols)) ;
    print( out=getNbr(0,img.cols-1,img.rows,img.cols));
    print(out= getNbr(img.rows-1,img.cols-1,img.rows,img.cols));
    print(out=getNbr(img.rows-1,0,img.rows,img.cols));
    */


    //img2=iScale(img,0.5,0.5);
    /*fname+=".tif";
    imwrite("orig_"+fname, img);
    imwrite("mine_"+fname, res);
    imwrite("ocv_"+fname, ores);
    */
    return 0;
} // end main




/*
hconcat(M1,M2,HM); // horizontal concatenation
 vconcat(M1,M2,VM); // vertical   concatenation

 //Create the rectangle
cv::Rect roi(10, 20, 100, 50);
//Create the cv::Mat with the ROI you need, where "image" is the cv::Mat you want to extract the ROI from
cv::Mat image_roi = image(roi)



*/


