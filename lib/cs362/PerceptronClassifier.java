package cs362;

import java.util.HashMap;
import java.util.List;

/**
 * Created by elric on 1/28/2016.
 */
public class PerceptronClassifier extends Predictor {

    private HashMap<Integer, Double> weightMap;
    private double learningRates;
    private List<Instance> trainingData;
    private int learningIterations;

    // Constructor with command line parameters.
    public PerceptronClassifier() {
        if (CommandLineUtilities.hasArg("online_learning_rate"))
            this.learningRates = CommandLineUtilities.getOptionValueAsFloat("online_learning_rate");
        if (CommandLineUtilities.hasArg("online_training_iterations"))
            this.learningIterations = CommandLineUtilities.getOptionValueAsInt("online_training_iterations");
    }

    public int getLearningIterations() {
        return learningIterations;
    }

    public void setLearningIterations(int learningIterations) {
        this.learningIterations = learningIterations;
    }

    public double getLearningRates() {
        return learningRates;
    }

    public void setLearningRates(double learningRates) {
        this.learningRates = learningRates;
    }

    @Override
    public void train(List<Instance> instances) {
        initialize(instances);

        for (int i = 0; i < learningIterations; i++) {
            for (Instance instance : trainingData) {
                int yi = (int) instance.getLabel().getValue();
                FeatureVector fv = instance.getFeatureVector();

                double wDotX = 0;

                for (int index : fv.getKeySet()) {
                    wDotX += weightMap.get(index) * fv.get(index);
                }
                // Compute the initial prediction.
                int yiPrediction;
                if (wDotX >= 0)
                    yiPrediction = 1;
                else
                    yiPrediction = -1;
                if (yi != yiPrediction) {
                    // If the prediction is wrong update the weight.
                    updateWeight(yi, fv, getLearningRates());
                }
            }
        }
    }

    // Initialize the data as given format and weights as 0.
    private void initialize(List<Instance> instances) {
        updateTrainingData(instances);

        int maxFeatureIndex = 0;
        for (Instance instance : instances) {
            for (int index : instance.getFeatureVector().getKeySet()) {
                if (index > maxFeatureIndex) {
                    maxFeatureIndex = index;
                }
            }
        }

        for (int i = 0; i < maxFeatureIndex; i++) {
            weightMap.put(i, 0.0);
        }
    }

    // Initialize data as -1 and 1.
    private void updateTrainingData(List<Instance> instances) {
        for (Instance instance : instances) {
            if (instance.getLabel().getValue() == 0.0) {
                trainingData.add(new Instance(instance.getFeatureVector(), new ClassificationLabel(-1)));
            } else {
                trainingData.add(new Instance(instance.getFeatureVector(), new ClassificationLabel(1)));
            }
        }
    }

    private void updateWeight(int yi, FeatureVector fv, double learningRates) {
        // update weights by adding given label data.
        for (int index : fv.getKeySet()) {
            weightMap.put(index, weightMap.get(index) + learningRates * yi * fv.get(index));
        }
    }

    @Override
    public Label predict(Instance instance) {
        double wDotX = 0;
        FeatureVector fv = instance.getFeatureVector();
        for (int index : fv.getKeySet()) {
            if (!weightMap.containsKey(index))
                continue;

            double weight = weightMap.get(index);
            wDotX += fv.get(index) * weight;
        }

        int yiPrediction;
        if (wDotX >= 0)
            yiPrediction = 1;
        else
            yiPrediction = 0;

        return new ClassificationLabel(yiPrediction);
    }
}
