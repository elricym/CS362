package cs362.nonlinear.kernel;

import cs362.FeatureVector;

/**
 * Created by elric on 2/4/2016.
 */
public class GaussianKernel extends KernelLogisticRegression {
    @Override
    protected double kernelFunction(FeatureVector featureVector1, FeatureVector featureVector2) {
        double norm = 0;
        // Calculate the dot product of two feature vectors.
        for (int i : featureVector1.getKeySet()) {
            double distance = (featureVector1.get(i) - featureVector2.get(i));
            norm += distance * distance;
        }
        return Math.exp(-1 * norm / (2 * gaussian_kernel_sigma * gaussian_kernel_sigma));
    }
}
