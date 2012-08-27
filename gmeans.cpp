#include <cv.h>
#include <highgui.h>
#include <queue>
#include <vector>
#include <algorithm>
using namespace std;

//reference: http://mathworld.wolfram.com/Erf.html
float erf(float v)
{
    float v2 = v*v;
    float v3 = v2* v;
    return (v-v3/3+v2*v3/10)*2/1.7724538357881143;
}
// N(0, 1) cumulative distribution function
float cdf(float v)
{
    return (1+ (erf(v/1.4142135623730951)))/2;
}
// test whether the distribution of data fits gaussian distribution
// input :
//      data, every row reprents a sample
// output: return row number of first cluster, only can be 1 or 2
//         if return value equals data->rows, clusters can be used to classify data 
// all data using float 
int fitGaussian(CvMat* data,CvMat* clusters)
{
    // if(data->type != CV_32FC1){
        // printf("Error: data type incorrect\n");
        // return NULL;
    // }
    if(data->rows <=1){
        return 1;
    }
    // two centers of data
    CvMat* centers = cvCreateMat(2,data->cols,data->type);
    // initialize the two centers 
    
    // find centers
    cvKMeans2( data, 2, clusters,
                cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ),
                10, 0, 0, centers, 0 );

    int kindCounts[2] ={0};
    
    for(int i =0 ; i < data->rows ; i++){
        int cluster = clusters->data.i[i];
        kindCounts[cluster]++;
    }

    float* ptr[2] = {(float*)centers->data.ptr,(float*)(centers->data.ptr + centers->step)};
    // connector, a vector that connect the two centers
    CvMat* connector = cvCreateMat(1,data->cols,CV_32FC1);
    // module of connector vector
    float connectorModule = 0;
    for(int col = 0; col < data->cols; col ++){
        float v = ptr[0][col] - ptr[1][col];
        connector->data.fl[col] = v;
        connectorModule += v * v;
        //printf("connector[%d] = %f\n",col,v);
    }
    if(connectorModule == 0){
        connectorModule = 0.0000001;
    }
    //printf("connectorModule = %f\n",connectorModule);
    // project data to connector
    CvMat* projection = cvCreateMat(data->rows,1,CV_32FC1);
    cvZero(projection);
    for(int r = 0; r < data->rows; r++){
        float* ptr = (float*)(data->data.ptr+ r * data->step);
        for(int c = 0; c < data->cols; c++){
            projection->data.fl[r] += ptr[c] * connector->data.fl[c];
        }
        projection->data.fl[r] /= connectorModule;
    }
    // sort projecion
    sort(projection->data.fl,projection->data.fl+projection->rows);
    
    // transform projection so that it has mean 0 and variance 1
    CvScalar avg;
    CvScalar stdDev;
    cvAvgSdv(projection,&avg,&stdDev);
    //printf("projection average = %f\n",avg.val[0]);
    //printf("projection stdDev = %f\n",stdDev.val[0]);
    // cumulative distribution function
    CvMat* zMat = cvCreateMat(data->rows,1,CV_32FC1);
    cvZero(zMat);
    for(int r = 0; r < data->rows; r++){
        projection->data.fl[r] = (projection->data.fl[r] - avg.val[0])/stdDev.val[0];        
        zMat->data.fl[r] = cdf(projection->data.fl[r]);
        //printf("zMat->data.fl[%d] = %f\n",r,zMat->data.fl[r]);
    }
    // statistic
    int n = data->rows;
    float A2Z = 0;
    for(int r = 0; r < data->rows; r++){
        A2Z += 2 * r * (log(zMat->data.fl[r]) + log(1 - zMat->data.fl[n - 1 - r]));
    }
    A2Z /= -n;
    A2Z -= n;
    // correction
    A2Z = A2Z*( 1 + 4./n - 25./(n * n));
    
    //printf("data->rows for now = %d\n",n);
    //printf("A2Z = %f\n",A2Z);
    
    const float significantLevel = 3;//0.0001;
    int ret = data->rows;
    if(abs(A2Z)>significantLevel){// does not fit
        ret = kindCounts[0];
    }
    // memory
    cvReleaseMat(&connector);
    cvReleaseMat(&projection);
    cvReleaseMat(&zMat);
    cvReleaseMat(&centers);
    
    return ret;
}

// G-means
// input: data to cluster, k is the initial cluster number
// output:
//      return the cluster number
//      clusters,similar to clusters of cvKMeans2 
int gmeans(CvMat* data,CvMat* clusters,int k = 1)
{
    queue<CvMat*> matsToCheck;
    vector<CvMat*> matsChecked;
    CvMat* tmat = cvCloneMat(data);
    matsToCheck.push(tmat);
    while(!matsToCheck.empty()){
        CvMat* currentSet = matsToCheck.front();
        matsToCheck.pop();
        CvMat* localClusters = cvCreateMat(currentSet->rows,1,CV_32SC1);
        int n = fitGaussian(currentSet,localClusters);
        if(n != currentSet->rows){
            CvMat* dataPart[2] = {
                cvCreateMat(n,currentSet->cols,CV_32FC1),
                cvCreateMat(currentSet->rows - n,currentSet->cols,CV_32FC1)
            };
            int index[2] = {0};
            int step = currentSet->step;
            for(int i =0 ; i <currentSet->rows; i++ ){                
                int c = localClusters->data.i[i];
                memcpy(dataPart[c]->data.ptr + index[c] * step,currentSet->data.ptr + i * step,step);
                index[c]++;
            }
            
            matsToCheck.push(dataPart[0]);
            matsToCheck.push(dataPart[1]);
            cvReleaseMat(&currentSet);
        }else{
            matsChecked.push_back(currentSet);
        }
        cvReleaseMat(&localClusters);
    }
    
    uchar* ptr = data->data.ptr;
    int index =0;
    for(int i =0 ; i < matsChecked.size();i++){
        int size = matsChecked[i]->step * matsChecked[i]->rows;
        memcpy(ptr,matsChecked[i]->data.ptr,size);
        ptr += size;
        for(int r =0; r < matsChecked[i]->rows; r ++){
            clusters->data.i[index +r] = i;
        }
        index += matsChecked[i]->rows;
        cvReleaseMat(&matsChecked[i]);
    }
    
    return matsChecked.size();
}

int main( int argc, char** argv )
{
    #define MAX_CLUSTERS 10
    CvScalar color_tab[MAX_CLUSTERS];
    IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
    CvRNG rng = cvRNG(-1);
    CvPoint ipt;

    color_tab[0] = CV_RGB(255,0,0);
    color_tab[1] = CV_RGB(0,255,0);
    color_tab[2] = CV_RGB(0,0,255);
    color_tab[3] = CV_RGB(255,255,0);
    color_tab[4] = CV_RGB(255,0,255);
    color_tab[5] = CV_RGB(0,255,255);
    color_tab[6] = CV_RGB(255,255,255);
    color_tab[7] = CV_RGB(0,128,255);
    color_tab[8] = CV_RGB(128,0,255);
    color_tab[9] = CV_RGB(255,128,0);

    cvNamedWindow( "clusters", 1 );

    for(;;)
    {
        char key;
        int k, cluster_count = cvRandInt(&rng)%MAX_CLUSTERS + 1;
        int i, sample_count = cvRandInt(&rng)%1000 + 1;
        CvMat* points = cvCreateMat( sample_count, 2, CV_32FC1 );
        CvMat* clusters = cvCreateMat( sample_count, 1, CV_32SC1 );
        cluster_count = MIN(cluster_count, sample_count);

        printf("cluster_count : %d\n",cluster_count);
        printf("sample_count : %d\n",sample_count);
        
        cvZero(clusters);
        /* generate random sample from multigaussian distribution */
        for( k = 0; k < cluster_count; k++ )
        {
            CvPoint center;
            CvMat point_chunk;
            center.x = cvRandInt(&rng)%img->width;
            center.y = cvRandInt(&rng)%img->height;
            cvGetRows( points, &point_chunk, k*sample_count/cluster_count,
                       k == cluster_count - 1 ? sample_count :
                       (k+1)*sample_count/cluster_count, 1 );

            cvRandArr( &rng, &point_chunk, CV_RAND_NORMAL,
                       cvScalar(center.x,center.y,0,0),
                       cvScalar(img->width*0.1,img->height*0.1,0,0));
        }

        /* shuffle samples */
        for( i = 0; i < sample_count/2; i++ )
        {
            CvPoint2D32f* pt1 = (CvPoint2D32f*)points->data.fl + cvRandInt(&rng)%sample_count;
            CvPoint2D32f* pt2 = (CvPoint2D32f*)points->data.fl + cvRandInt(&rng)%sample_count;
            CvPoint2D32f temp;
            CV_SWAP( *pt1, *pt2, temp );
        }

        cvKMeans2( points, cluster_count, clusters,
                cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ),
                5, 0, 0, 0, 0 );
        
        cvZero( img );

        for( i = 0; i < sample_count; i++ )
        {
            int cluster_idx = clusters->data.i[i];
            ipt.x = (int)points->data.fl[i*2];
            ipt.y = (int)points->data.fl[i*2+1];
            //printf("cluster_idx = %d; point (%d,%d)\n",cluster_idx,ipt.x,ipt.y);
            cvCircle( img, ipt, 2, color_tab[cluster_idx%MAX_CLUSTERS], CV_FILLED, CV_AA, 0 );
        }

        cvShowImage( "kmeans", img );
        cvSaveImage( "kmeans.jpg", img );
        ///////////////////////////////////////////////////////////////////
        printf("cluster number by gmeans: %d\n",gmeans(points,clusters));
        
        cvZero( img );

        for( i = 0; i < sample_count; i++ )
        {
            int cluster_idx = clusters->data.i[i];
            ipt.x = (int)points->data.fl[i*2];
            ipt.y = (int)points->data.fl[i*2+1];
            //printf("cluster_idx = %d; point (%d,%d)\n",cluster_idx,ipt.x,ipt.y);
            cvCircle( img, ipt, 2, color_tab[cluster_idx%MAX_CLUSTERS], CV_FILLED, CV_AA, 0 );
        }

        cvShowImage( "gmeans", img );
        cvSaveImage( "gmeans.jpg", img );

        cvReleaseMat( &points );
        cvReleaseMat( &clusters );
        
        key = (char) cvWaitKey(0);
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }

    cvDestroyWindow( "image" );
    cvReleaseImage(&img);
    return 0;
}
