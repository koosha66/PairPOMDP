/* 
 * File:   POMDP.h
 * Author: koosha
 *
 * Created on November 25, 2011, 4:16 PM
 */

#ifndef POMDP_H
#define	POMDP_H

class POMDP {
public : 
    int nrStates ;
    int nrActions ;
    int nrObservations ;
    float gamma ;
    float ***reward ;
    float ***observation ;
    float ***transition ;
    float *start ;
    int **states_actions ;
    int **states_observations ;
    POMDP(char* file_name) ;
    ~POMDP() ;
    
    
};

#endif	/* POMDP_H */

