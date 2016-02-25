package cs362.nonlinear.kernel;

import cs362.*;

import java.util.List;

/**
 * Created by elric on 2/2/2016.
 */
public abstract class KernelLogisticRegression extends Predictor {

    protected double polynomial_kernel_exponent;
    protected double gaussian_kernel_sigma;
    private double[][] gramMatrix;
    private double[] cachedAlphaKernel;
    private double[] alpha;
    private int iterations;
    private double learningRate;
    private List<Instance> trainingData;

    public KernelLogisticRegression() {

        // Load kernel parameters from command line.
        learningRate = 0.01;
        if (CommandLineUtilities.hasArg("gradient_ascent_learning_rate"))
            learningRate =
                    CommandLineUtilities.getOptionValueAsFloat("gradient_ascent_learning_rate");

        iterations = 5;
        if (CommandLineUtilities.hasArg("gradient_ascent_training_iterations"))
            iterations =
                    CommandLineUtilities.getOptionValueAsInt("gradient_ascent_training_iterations");

        polynomial_kernel_exponent = 2;
        if (CommandLineUtilities.hasArg("polynomial_kernel_exponent"))
            polynomial_kernel_exponent = CommandLineUtilities.getOptionValueAsFloat("polynomial_kernel_exponent");

        gaussian_kernel_sigma = 1;
        if (CommandLineUtilities.hasArg("gaussian_kernel_sigma"))
            gaussian_kernel_sigma = CommandLineUtilities.getOptionValueAsFloat("gaussian_kernel_sigma");
    }

    @Override
    public void train(List<Instance> instances) {
        trainingData = instances;
        initialize(instances.size());
        computeGramMatrix(instances);
        for (int i = 0; i < iterations; i++) {
            computeAlphaKernel(instances.size());

            // For each alpha in the vector, update the value with learning rate and gradient.
            for (int k = 0; k < instances.size(); k++) {
                double gradientAlphaK = computeGradient(instances, k);
                alpha[k] += learningRate * gradientAlphaK;
            }
        }
    }

    @Override
    public Label predict(Instance instance) {
        double probability = computeProbability(instance, alpha, trainingData);
        if (probability >= 0.5) {
            return new ClassificationLabel(1);
        } else {
            return new ClassificationLabel(0);
        }
    }

    // Initialize matrices with data size.
    private void initialize(int dataSize) {
        gramMatrix = new double[dataSize][dataSize];
        cachedAlphaKernel = new double[dataSize];
        alpha = new double[dataSize];
    }

    private double linkFunction(double z) {
        return 1 / (1 + Math.exp(-1 * z));
    }


    // Abstract method to be implemented by 3 different kernel functions.
    protected abstract double kernelFunction(FeatureVector featureVector1, FeatureVector featureVector2);

    // Compute and store gram matrix from training data.
    private void computeGramMatrix(List<Instance> instances) {
        for (int i = 0; i < instances.size(); i++) {
            for (int j = 0; j < instances.size(); j++) {
                gramMatrix[i][j] += kernelFunction(
                        instances.get(i).getFeatureVector(), instances.get(j).getFeatureVector());
            }
        }
    }

    // Compute and cache the product of alpha vector and kernels.
    private void computeAlphaKernel(int dataSize) {
        for (int i = 0; i < dataSize; i++) {
            double temp = 0;
            for (int j = 0; j < dataSize; j++) {
                temp += alpha[j] * gramMatrix[j][i];
            }
            cachedAlphaKernel[i] = temp;
        }
    }

    // Compute gradient of alpha for given k.
    private double computeGradient(List<Instance> instances, int k) {
        double gradient = 0;
        for (int i = 0; i < instances.size(); i++) {
            double value = instances.get(i).getLabel().getValue();
            if (value == 1.0) {
                gradient += linkFunction(-1 * cachedAlphaKernel[i]) * gramMatrix[i][k];
            } else {
                gradient += linkFunction(cachedAlphaKernel[i]) * (-1 * gramMatrix[i][k]);
            }
        }
        return gradient;
    }

    // Compute the probability of y for making prediction.
    private double computeProbability(Instance instance, double[] alpha, List<Instance> trainingData) {
        double alphaKernel = 0;
        for (int j = 0; j < trainingData.size(); j++) {
            alphaKernel += alpha[j] *
                    kernelFunction(trainingData.get(j).getFeatureVector(), instance.getFeatureVector());
        }
        // Link function result of the product of alpha and kernel.
        return linkFunction(alphaKernel);
    }
}
