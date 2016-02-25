package cs362;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * Created by elric on 1/28/2016.
 */
public class NaiveBayesClassifier extends Predictor {

    private List<Instance> trainingData;
    private HashSet<Integer> featureSet;
    private HashMap<Label, HashMap<Integer, HashMap<Double, Integer>>> valueCountInLabelAndFeature;
    private HashMap<Label, Double> priorProbability;
    private HashMap<Label, HashMap<Integer, Double>> conditionalFeatureProbability;
    private HashMap<Label, Double> likelihood;
    private double lambda;

    public NaiveBayesClassifier() {
        trainingData = new ArrayList<>();
        valueCountInLabelAndFeature = new HashMap<>();
    }

    public void initialize(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public void train(List<Instance> instances) {
        convertFeaturesIntoBinary(instances);
        countValueFrequency(instances);
        HashMap<Label, Integer> labelCount = countLabelFrequency(trainingData);
        // Calculate the prior probability and feature probabilities as model.
        priorProbability = computePriorProbability(trainingData, lambda, labelCount);
        conditionalFeatureProbability = computeConditionalFeatureProbability(lambda, priorProbability, labelCount);
    }

    // Convert double features into binary.
    private void convertFeaturesIntoBinary(List<Instance> instances) {

        for (Instance instance : instances) {
            FeatureVector fv = instance.getFeatureVector();
            FeatureVector converted = new FeatureVector();
            for (int index : fv.getKeySet()) {
                featureSet.add(index);
                if (fv.get(index) < 0.5)
                    converted.add(index, 0);
                else
                    converted.add(index, 1);
            }
            this.trainingData.add(new Instance(converted, instance.getLabel()));
        }
    }

    // Simply count the frequency of every possible feature values.
    private void countValueFrequency(List<Instance> instances) {
        for (Instance instance : instances) {
            FeatureVector fv = instance.getFeatureVector();
            for (int index : fv.getKeySet()) {

                double value = fv.get(index);
                HashMap<Integer, HashMap<Double, Integer>> valueCountOfFeature;
                HashMap<Double, Integer> valueAndCount;

                if (valueCountInLabelAndFeature.containsKey(instance.getLabel())) {
                    valueCountOfFeature = valueCountInLabelAndFeature.get(instance.getLabel());
                    if (valueCountOfFeature.containsKey(index)) {
                        valueAndCount = valueCountOfFeature.get(index);
                        if (valueAndCount.containsKey(value)) {
                            valueAndCount.put(value, valueAndCount.get(value) + 1);
                        } else {
                            valueAndCount.put(value, 1);
                        }
                    } else {
                        valueAndCount = new HashMap<>();
                        valueAndCount.put(value, 1);
                    }
                } else {
                    valueCountOfFeature = new HashMap<>();
                    valueCountInLabelAndFeature.put(instance.getLabel(), valueCountOfFeature);
                    valueAndCount = new HashMap<>();
                    valueAndCount.put(value, 1);
                }
                valueCountOfFeature.put(index, valueAndCount);
            }
        }
    }

    // Simply count the frequency of labels.
    private HashMap<Label, Integer> countLabelFrequency(List<Instance> instances) {
        HashMap<Label, Integer> labelCount = new HashMap<>();
        for (Instance instance : instances) {
            Label label = instance.getLabel();
            if (labelCount.containsKey(label)) {
                labelCount.put(label, labelCount.get(label) + 1);
            } else {
                labelCount.put(label, 1);
            }
        }
        return labelCount;
    }

    private HashMap<Label, Double> computePriorProbability(
            List<Instance> instances, double lambda, HashMap<Label, Integer> labelCount) {

        HashMap<Label, Double> priorProbability = new HashMap<>();
        // For each label, compute its probability and add lambda for smoothing.
        for (Label label : labelCount.keySet()) {
            double probability = (labelCount.get(label) + lambda) / (instances.size() + lambda * labelCount.size());
            priorProbability.put(label, probability);
        }
        return priorProbability;
    }

    private HashMap<Label, HashMap<Integer, Double>> computeConditionalFeatureProbability(
            double lambda, HashMap<Label, Double> priorProbability, HashMap<Label, Integer> labelCount) {
        HashMap<Label, HashMap<Integer, Double>> conditionalFeatureProbability = new HashMap<>();
        for (Label label : priorProbability.keySet()) {
            if (!conditionalFeatureProbability.containsKey(label)) {
                conditionalFeatureProbability.put(label, new HashMap<>());
            }

            // For each feature value, compute its probability and add lambda for smoothing.
            HashMap<Integer, HashMap<Double, Integer>> valueCountOfFeature = valueCountInLabelAndFeature.get(label);
            for (int index : valueCountOfFeature.keySet()) {
                double pXC;
                if (valueCountOfFeature.containsKey(1.0)) {
                    pXC = (valueCountOfFeature.get(index).get(1.0) + lambda) / (labelCount.get(label) + lambda);
                } else {
                    pXC = lambda / (labelCount.get(label) + lambda);
                }
                conditionalFeatureProbability.get(label).put(index, pXC);
            }
        }
        return conditionalFeatureProbability;
    }

    @Override
    public Label predict(Instance instance) {
        for (Label label : conditionalFeatureProbability.keySet()) {
            double logPriorProbability = Math.log(priorProbability.get(label));
            double logSumConditionalProbability = 0.0;
            FeatureVector fv = instance.getFeatureVector();
            for (int index : fv.getKeySet()) {
                // Skip features that not included in training data.
                if (!featureSet.contains(index))
                    continue;

                // Compute the summation of conditional probability.
                double value = fv.get(index);
                double pXiC = getPXiC(label, index, value);
                logSumConditionalProbability += Math.log(pXiC);
            }
            // Compute likelihood for each possible label.
            likelihood.put(label, logPriorProbability + logSumConditionalProbability);
        }
        return getPredictionByMaxLikelihood(likelihood);
    }

    // Compute the conditional probability for a single X.
    private double getPXiC(Label label, int index, double value) {
        double pXiC;
        if (value < 0.5) {
            pXiC = 1 - conditionalFeatureProbability.get(label).get(index);
        } else {
            pXiC = conditionalFeatureProbability.get(label).get(index);
        }
        return pXiC;
    }

    // Compute the prediction by check the maximum likelihood.
    private Label getPredictionByMaxLikelihood(HashMap<Label, Double> likelihood) {
        Label prediction = null;
        double maxLikelihood = 0.0;
        for (Label label : likelihood.keySet()) {
            if (likelihood.get(label) > maxLikelihood) {
                maxLikelihood = likelihood.get(label);
                prediction = label;
            }
        }
        return prediction;
    }

}
