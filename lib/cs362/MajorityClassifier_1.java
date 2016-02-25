package cs362;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

/**
 * Created by elric on 1/19/2016.
 */
public class MajorityClassifier_1 extends Predictor {

    private HashMap<FeatureVector, Label> model;

    public MajorityClassifier_1() {
        model = new HashMap<>();
    }

    @Override
    public void train(List<Instance> instances) {
        HashMap<FeatureVector, List<Integer>> labelMap = new HashMap<>();
        for (Instance instance : instances) {
            FeatureVector fv = instance.getFeatureVector();
            Label label = instance.getLabel();
            if (labelMap.containsKey(fv)) {
                labelMap.get(fv).add(Integer.parseInt(label.toString()));
            } else {
                List<Integer> labelList = new ArrayList<>();
                labelList.add(Integer.parseInt(label.toString()));
                labelMap.put(fv, labelList);
            }
        }
        for (FeatureVector fv : labelMap.keySet()) {
            List<Integer> labelList = labelMap.get(fv);
            Collections.sort(labelList);
            int maxCount = Collections.frequency(labelList, labelList.get(0));
            int currentNum = labelList.get(0);
            List<Integer> maxSet = new ArrayList<>();
            maxSet.add(currentNum);
            for (int i : labelList) {
                if (i == currentNum)
                    continue;
                else {
                    int freq = Collections.frequency(labelList, i);
                    if (maxCount == freq) {
                        maxSet.add(i);
                        currentNum = i;
                    } else if (maxCount < freq) {
                        maxSet.clear();
                        maxSet.add(i);
                        maxCount = freq;
                    } else {
                        currentNum = i;
                    }
                }
            }
            if (maxSet.size() == 1) {
                model.put(fv, new ClassificationLabel(maxSet.get(0)));
            } else {
                int rand = (int) (Math.random() * maxSet.size() + 1);
                model.put(fv, new ClassificationLabel(rand));
            }
        }
    }

    @Override
    public Label predict(Instance instance) {
        if (model.containsKey(instance.getFeatureVector()))
            return model.get(instance.getFeatureVector());
        else
            return new ClassificationLabel(0);
    }
}
