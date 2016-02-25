package cs362;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by elric on 1/19/2016.
 */
public class EvenOddClassifier extends Predictor {

    // Save the model generated from training.
    private HashMap<FeatureVector, Label> model;

    // Default constructor.
    public EvenOddClassifier() {
        model = new HashMap<>();
    }

    @SuppressWarnings("Duplicates")
    @Override
    public void train(List<Instance> instances) {
        for (Instance instance : instances) {
            FeatureVector featureVector = instance.getFeatureVector();
            int evenSum = 0;
            int oddSum = 0;
            for (Map.Entry<Integer, Double> entrySet : featureVector.entrySet()) {
                if (entrySet.getKey() % 2 == 0)
                    oddSum += entrySet.getValue();
                else
                    evenSum += entrySet.getValue();
            }
            if (evenSum >= oddSum)
                model.put(featureVector, new ClassificationLabel(1));
            else
                model.put(featureVector, new ClassificationLabel(0));
        }
    }

    @Override
    public Label predict(Instance instance) {
        FeatureVector featureVector = instance.getFeatureVector();
        int evenSum = 0;
        int oddSum = 0;
        for (Map.Entry<Integer, Double> entrySet : featureVector.entrySet()) {
            if (entrySet.getKey() % 2 == 0)
                oddSum += entrySet.getValue();
            else
                evenSum += entrySet.getValue();
        }
        if (evenSum >= oddSum)
            return new ClassificationLabel(1);
        else
            return new ClassificationLabel(0);
    }
}
