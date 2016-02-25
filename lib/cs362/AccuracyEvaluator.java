package cs362;

import java.util.List;

/**
 * Created by elric on 1/21/2016.
 * <p>
 * AccuracyEvaluator, extended from Evaluator, is the class to calculate
 * the accuracy of our predictions.
 */
public class AccuracyEvaluator extends Evaluator {
    @Override
    public double evaluate(List<Instance> instances, Predictor predictor) {
        double accuracy = 0.0;

        // Traversing instances and checking if the predictions meet data.
        for (Instance instance : instances) {
            Label prediction = predictor.predict(instance);
            Label data = instance.getLabel();

            // When predictions equals data, accuracy adds one.
            if (prediction.toString().equals(data.toString()))
                accuracy += 1.0;
        }
        return accuracy / instances.size();
    }
}
