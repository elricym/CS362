package cs362;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by elric on 1/21/2016.
 */
public class MajorityClassifier extends Predictor {


    public Label model;

    @Override
    public void train(List<Instance> instances) {

        // Set a hashmap to save labels and its amount.
        HashMap<Integer, Integer> labelCount = new HashMap<>();

        // CommonLabel is to save the most common label. Initialize it with the first item.
        int commonLabel = Integer.parseInt(instances.get(0).getLabel().toString());
        int commonCount = 1;
        labelCount.put(commonLabel, commonCount);

        // Set an array list to save possible most common labels that have a same number.
        List<Integer> commonSet = new ArrayList<>();
        commonSet.add(commonLabel);

        // Go through all the instances and count the number of labels.
        for (Instance instance : instances) {
            int currentLabel = Integer.parseInt(instance.getLabel().toString());

            // Check if this label occurred before.
            if (!labelCount.containsKey(currentLabel)) {
                // If not, add a new item in labelCount and then
                // check it is able to be a most common label.
                labelCount.put(currentLabel, 1);
                if (commonCount == 1)
                    commonSet.add(currentLabel);
            } else {
                // If occurred before, add the amount of it.
                labelCount.replace(currentLabel, labelCount.get(currentLabel) + 1);
                if (currentLabel == commonLabel)
                    commonCount++;
                else if (commonCount == labelCount.get(currentLabel)) {
                    commonSet.add(currentLabel);
                } else if (commonCount < labelCount.get(currentLabel)) {
                    // If the amount of this label goes beyond the most common one,
                    // clear the result set and add this label into it.
                    commonSet.clear();
                    commonSet.add(currentLabel);
                    commonCount = labelCount.get(currentLabel);
                }
            }
        }

        // Check the size of result set.
        if (commonSet.size() == 1)
            model = new ClassificationLabel(commonSet.get(0));
        else {
            // Randomly select a label from result set as result and save it.
            int rand = (int) (Math.random() * commonSet.size() + 1);
            model = new ClassificationLabel(rand);
        }
    }

    @Override
    public Label predict(Instance instance) {

        // Get the predicted result. Simply return the saved result.
        return model;
    }
}
