/* 
 * File:   POMDP.cpp
 * Author: koosha
 * 
 * Created on November 25, 2011, 4:16 PM
 */

#include "POMDP.h"
#include <stdio.h>

POMDP::POMDP(char* file_name) {
    int i, j,k ; 
    float max_transition , max_observation ;
    FILE* in = fopen(file_name , "r") ;
    fscanf (in,"%d",&nrStates) ;
    fscanf (in,"%d",&nrActions) ;
    fscanf (in,"%d",&nrObservations) ;
    fscanf (in,"%f",&gamma) ;
    reward = new float**[nrStates] ;
    for (i=0 ; i<nrStates ; i++)
    {
        reward [i] = new float*[nrStates] ;
        for (j=0 ; j<nrStates ; j++)
            reward[i][j]=new float[nrActions];      
    }
        
    for (i=0 ; i<nrActions ; i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrStates ; k++)
                fscanf (in,"%f",&reward[j][k][i]);
        
        
    observation = new float**[nrStates] ;
    for (i=0 ; i<nrStates ; i++)
    {
        observation [i] = new float*[nrActions] ;
        for (j=0 ; j<nrActions ; j++)
            observation[i][j]=new float[nrObservations];      
    }
        
    for (i=0 ; i<nrObservations ; i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrActions ; k++)
                fscanf (in,"%f",&observation[j][k][i]);
        
    
    transition = new float**[nrStates] ;
    for (i=0 ; i<nrStates ; i++)
    {
        transition [i] = new float*[nrStates] ;
        for (j=0 ; j<nrStates ; j++)
            transition[i][j]=new float[nrActions];      
    }
        
    for (i=0 ; i<nrActions ; i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrStates ; k++)
                fscanf (in,"%f",&transition[j][k][i]);
    
    start =new float[nrStates];
    for (i=0;i<nrStates ; i++)
        fscanf (in,"%f",&start[i]) ;   
    
    states_actions = new int*[nrStates] ;
    for (i=0; i < nrStates ; i++)
        states_actions [i] = new int[nrActions] ;
    

    for (i=0 ;i < nrStates ; i++)
        for (j=0 ; j<nrActions ; j++)
        {
            max_transition = transition[0][i][j] ;
            states_actions[i][j] = 0 ;
            for (k=1 ; k<nrStates ; k++)
                if (transition[k][i][j] > max_transition)
                {
                        max_transition = transition[k][i][j] ;
                        states_actions[i][j] = k ;
                }       
        }
    
    
    
    states_observations = new int*[nrStates] ;
    for (i=0; i < nrStates ; i++)
        states_observations [i] = new int[nrActions] ;
    

    for (i=0 ;i < nrStates ; i++)
        for (j=0 ; j<nrActions ; j++)
        {
            max_observation = observation[i][j][0] ;
            states_observations[i][j] = 0 ;
            for (k=1 ; k<nrObservations ; k++)
                if (observation[i][j][k] > max_observation)
                {
                        max_observation = observation[i][j][k] ;
                        states_observations[i][j] = k ;
                }       
        }
                       
}

POMDP::~POMDP() {
    int i, j ; 
    for (i=0; i < nrStates ; i++)
    {
        delete []states_observations [i] ;
        delete []states_actions [i] ;
    }
    delete []states_actions ;
    delete []states_observations ;
    
    for (i=0 ; i<nrStates ; i++)
    {
        for (j=0 ; j<nrStates ; j++)
        {
            delete []transition[i][j];
            delete []reward[i][j] ;
        }
        delete []transition[i];
        delete []reward[i] ;
    }
    delete []transition ;   
    delete []reward ;

    for (i=0 ; i<nrStates ; i++)
    {
        for (j=0 ; j<nrActions ; j++)
            delete []observation[i][j];      
        delete []observation[i];
    }
    delete [] observation ;
}


