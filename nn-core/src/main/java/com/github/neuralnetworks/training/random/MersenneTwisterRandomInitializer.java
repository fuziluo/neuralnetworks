package com.github.neuralnetworks.training.random;

import java.util.Random;

import org.uncommons.maths.random.MersenneTwisterRNG;


/**
 * Mersenne twister random initializer
 */
public class MersenneTwisterRandomInitializer extends RandomInitializerImpl {

    private static final long serialVersionUID = 1L;

    public MersenneTwisterRandomInitializer() {
	super(new MersenneTwisterRNG());
    }
    

    public MersenneTwisterRandomInitializer(float start, float end) {
//    	super(new MersenneTwisterRNG(), start, end);
    	super(new Random(0), start, end); //fix the seed
    }
}
