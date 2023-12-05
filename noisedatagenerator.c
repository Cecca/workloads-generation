#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_gauss 1000 //
float *gauss(float ex,float dx,int n_point)
{ 
	time_t t;
	int i;
	float *mem1;
	mem1 = (float *)malloc(n_point*sizeof(float));
	srand((unsigned)time(&t));
	for(i=0;i<n_point;i++)
		mem1[i]=(sqrt(-2*log((float)rand()/32768))*cos((float)rand()/32768*2*3.1415926))*sqrt(dx)+ex;
	return(mem1);
}


int main(int argc, char **argv)
{
    FILE * ifile;
    char * dataset = argv[1];
	char * outpputfile = argv[2];
int newdatasetnumber=	atoi(argv[3]);	
int tssize=atoi(argv[4]);
unsigned long int originaldatasetsize=atoi(argv[5]);
float noiselevel=	atof(argv[6]);
    float *ts_buffer     = malloc(sizeof(float) * tssize);
    FILE * full_file = fopen(outpputfile, "wb");
//FILE * datafile = fopen(newdatafile, "a+");
    ifile = fopen (dataset,"rb");
    time_t t;
    srand((unsigned) time(&t));
    unsigned long p;

    for (int  i = 0; i < newdatasetnumber; i++)
    {
	
         p =(unsigned long)rand()%originaldatasetsize;
		printf("Query %d, Offset = %ld\n",(i+1),p);
		fseek(ifile, p * tssize*sizeof(float), SEEK_SET); 
		fread(ts_buffer, sizeof(float), tssize, ifile);
	for(int j=0;j<tssize;j++)
	{


		int a =rand()%999+1;
		float f=(float) a;
		f=f/1000.0f;
		float c=sqrt(-2.0*log(f));
		int a2 =rand()%1000;
		float f2=(float) a2;
		f2=f2/1000.0f;
		float b=2*3.1415926*f2;
		//float nosie=(float)c*cos(b)*sqrt(noiselevel);
		float nosie=(float)c*cos(b)*noiselevel;
		printf("old value = %g", ts_buffer[j]);
		ts_buffer[j]=ts_buffer[j]+nosie;
		printf("           new value =%g\n", ts_buffer[j]);


	}
	float sum=0.0f;
	float stdd=0.0f;
		for(int j=0;j<tssize;j++)
	{
		sum=sum+ts_buffer[j];

	}
	float average=sum/tssize;
	for(int j=0;j<tssize;j++)
	{
		stdd=stdd+(ts_buffer[j]-average)*(ts_buffer[j]-average);
	}
	float stdev=sqrt(stdd/tssize);

	for(int j=0;j<tssize;j++)
	{
		ts_buffer[j]=(ts_buffer[j]-average)/stdev;

	}
	fwrite(ts_buffer,sizeof(float), tssize, full_file);
    }

    fclose(ifile);
    fclose(full_file);
    free(ts_buffer);
    return 0;
}







