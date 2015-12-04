/*
 * mainDriver.cpp
 *
 *  Created on: Oct 8, 2015
 *      Author: afarasat
 */
#include <iostream>
#include <string.h>
#include <math.h>
#include <vector>
#include <cstdlib>
#include "solution.h"
#include "dataPreparation.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>

using namespace std;




int main(int argc, char* args[]) {
	// *****************************
	// Parallel Setup
	int NUM_PROCS, NUM_THREADS;
	//NUM_PROCS = omp_get_num_procs(); // number of processes requested
	NUM_PROCS = atoi(args[4]);
	//NUM_PROCS = (int)(args[0]);
	int Q = atoi(args[1]);
	int N =atoi(args[2]);
	int K = atoi(args[3]);
	int runMax = atoi(args[5]);
	NUM_THREADS = NUM_PROCS;
	omp_set_num_threads(NUM_THREADS); 
	
	// Parameters Setting
	float mu = 1.0; // This represent parameter of l2 regularization
	float init_mu = mu;
	float lambda = 1.0; // This represent parameter of l1 regularization
	float gamma = 1.0; // This represent parameter of Frobenius norm
	float init_gamma = gamma;
	static int numOfIteration = 2000; // Number of Iteration of main optimization loop
	double sparcityParameter =0.05 ;
	float stepSize; // The stepsize in the optimization process 
	float init_stepSize = stepSize;
	// number of processes requested

	
	// Generating Random Indexes for whole otimization
	
	//*****************************
	int ** data; // full data
	int ** observation; // observation = data-missing value
	std::vector<double> W;
	std::vector<double> C;
	int bestItr;
	long currObj, bestObj, initObj;
	double bestTime;
	bool netFlix = false;
	int bestITR [runMax];
	long bestOBJ[runMax];
	double bestTIME[runMax];
	double totTIME[runMax];
	long improvObj[runMax];
	if (netFlix){

	}else{
		dataPreparation dataObj(Q,N,K);
		dataObj.initialization();
		data = dataObj.getMainData();
		observation = dataObj.getObservation();
	}
	int index;
    	double timeStart,timeEnd;
	int num_rand = numOfIteration*NUM_PROCS;
	 // Generating Random Indexes for whole otimization
	std::vector<double> I_index(num_rand);
	std::vector<double> K_index(num_rand);
	std::vector<double> J_index(num_rand);
	double timeS_Obj, timeE_Obj;
	int numRun;
	std::ofstream myFile;
	try{
		myFile.open("Results//Q"+std::to_string(Q)+"N"+std::to_string(N)+"K"+std::to_string(K)+".txt");
	}
	catch (exception& e){
		std::cout << "c++11 is required to use std::to_string" << std::endl;
	}
	int currentIteration;
	int counter;	
	cout<<"****************"<<" OBSERVATION "<<endl;
	for (numRun = 0; numRun < runMax; numRun++){
		
		mu = init_mu ;
        	gamma = init_gamma;
        	stepSize = init_stepSize; // The stepsize in the optimization process
		timeStart = omp_get_wtime();
		#pragma omp parallel shared(I_index,K_index,J_index,numOfIteration,NUM_PROCS)
		{
		srand(time(NULL)+omp_get_thread_num()); 
		#pragma omp for private(index) 	
			for (index = 0; index < num_rand; index++){
					I_index[index] =(rand() % (Q));
					K_index[index] = (rand() % (K));
					J_index[index] = (rand() % (N));
			}
		}

		solution solutionObj(observation,  NUM_PROCS,sparcityParameter, Q, N, K);
		W = solutionObj.getW();
		C = solutionObj.getC();	
		timeS_Obj = omp_get_wtime();
		currObj =  solutionObj.objectiveFunction(lambda, mu,gamma);
		initObj = currObj;
		myFile << "Run: " << numRun << std::endl;
		//myFile << "Iteration" << "\t" << "Objective Function" << std::endl;
		myFile << "Initial Value" << "\t" << currObj << std::endl;
		//cout << "Objective function: " << currObj << endl;
		bestObj = currObj;
		bestItr = 0;
		bestTime = timeStart;
		timeE_Obj = omp_get_wtime();
		std::cout << "Time Objective Function: " << timeE_Obj - timeS_Obj <<  std::endl;
		cout<<"****************"<<" Optimization Started..."<<endl;
		currentIteration = 0;
		stepSize = 2/(currentIteration+2);
		counter = 1;
		while (currentIteration < numOfIteration){
			//NUM_PROCS = 1;
			omp_set_num_threads(NUM_PROCS);
			int tid;
			#pragma omp parallel shared(W,C)
			{ 
			int iW, kW, kC, jC;
			//#pragma omp for schedule(dynamic) private(iW, kW, kC, jC,procIterator)
			//for (procIterator = 0; procIterator < NUM_PROCS; procIterator++){
				tid = omp_get_thread_num();
				//std::cout<< "ThreadID: " << tid << " rand: " << I_index[currentIteration*NUM_PROCS+tid]<< std::endl;
				iW = I_index[currentIteration*NUM_PROCS+tid];
				kW = K_index[currentIteration*NUM_PROCS+tid];
				jC = J_index[currentIteration*NUM_PROCS+tid];
				//iW =(rand() % (Q));
				//kW = (rand() % (K));
				//printf("process id: %d Iterator: %d iW: %d kW: %d\n",tid,procIterator,iW,kW); 
			
				W[iW*K+kW] = solutionObj.updateWij(iW,kW,mu,stepSize);
				//W[iW]  = solutionObj.updateRowWij(iW,mu,stepSize);
				kC = kW; //jC = (rand() % (N));
				C[kC*N+jC] = solutionObj.updateCij(kC,jC,gamma,stepSize);
				//C[jC] = solutionObj.updateColumnCij(jC,gamma,stepSize);
				//cout << "C[i][j]: " << C[kC][jC] << endl;
			//}
			}
			//#pragma omp barrier	
			currObj = solutionObj.objectiveFunction(lambda, mu,gamma);
		//	myFile << currentIteration  << "\t" << currObj << std::endl;
			cout << "Iteration: " << currentIteration << ", Objective function: " << currObj << endl;
			if (currObj < bestObj){
				bestObj = currObj;
				bestItr = currentIteration;
				bestTime = omp_get_wtime();
				bestITR[numRun] = bestItr;
				bestOBJ[numRun] = bestObj;
				bestTIME[numRun] = bestTime - timeStart;
				improvObj[numRun] = bestObj - initObj;
			}
			stepSize = 2/(currentIteration+2);	
		//	stepSize /= 1.1;
		//	mu = mu/1.05;
		//	gamma = gamma/1.05;
			/*
			if(currentIteration % 200 == 0 && currentIteration != 0 ){
				counter++;
				stepSize = ((float)1/(float)counter) * init_stepSize;
				mu = ((float)1/(float)counter) * init_mu;
				gamma = ((float)1/(float)counter) * init_gamma;
			} 
			*/
			currentIteration++;
			
		}
		//std::cout << "mu:" << mu << " step:" << stepSize << "gamma:" << gamma << std::endl;
		timeEnd =omp_get_wtime();
		myFile << "Best Objective Function: " << bestObj << "\t" << "Best Iteration: " << bestItr << "\t" << "Best Time" << bestTime-timeStart << std::endl;
		myFile << "Num of Procs: " <<  NUM_PROCS  << "\t" << " Total time: " << timeEnd-timeStart << std::endl;
		std::cout << "Num of Procs: " <<  NUM_PROCS  <<" Total time: " << timeEnd-timeStart << std::endl;
		totTIME[numRun] = timeEnd-timeStart;
		solutionObj.~solution();
		W.erase(W.begin(),W.end());
		C.erase(C.begin(),C.end());
	}
	double sumBestOBJ = 0.0;
	double sumbestTime = 0.0;
	double sumTotTIME = 0.0;
	int sumBestITR = 0;
	double sumImprov = 0.0;
	myFile << "Best Objective Function" << "\t" << "Best Iteration" << "\t" << "Best Time" << "\t" << "Total Time"<< "\t" << "Improvement" << std::endl;
	for (int i = 0; i < runMax; i++){
		sumBestOBJ +=  bestOBJ[i];
		sumBestITR +=  bestITR[i];
		sumbestTime += bestTIME[i];
		sumTotTIME += totTIME[i];
		sumImprov += improvObj[i]; 
		myFile << bestOBJ[i]  << "\t" << bestITR[i] << "\t" << bestTIME[i] << "\t" << totTIME[i] << "\t" << improvObj[i] << std::endl;
	}
	myFile << "Average Objective Function" << "\t" << "Average Iteration" << "\t" << "Average Best Time" << "\t" << "Average Total Time"<< "\t"<<"Average Improvement"<< std::endl;
	myFile <<(float) sumBestOBJ/(float)runMax  << "\t" <<(float)sumBestITR/(float)runMax << "\t" << (float)sumbestTime/(float)runMax << "\t" << (float)sumTotTIME/(float)runMax << "\t"<<(float)sumImprov/(float)runMax <<std::endl;
	myFile.close();
return -1;
}
