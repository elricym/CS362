package cs362.nonlinear.kernel;

import cs362.FeatureVector;

/**
 * Created by elric on 2/4/2016.
 */
public class PolynomialKernel extends KernelLogisticRegression {

    @Override
    protected double kernelFunction(FeatureVector featureVector1, FeatureVector featureVector2) {
        double kernel = 0.0;

        // Calculate the dot product of two feature vectors.
        for (int i : featureVector1.getKeySet()) {
            for (int j : featureVector2.getKeySet()) {
                kernel += featureVector1.get(i) * featureVector2.get(j);
            }
        }
        return Math.pow(1 + kernel, polynomial_kernel_exponent);
    }
}
