package cs362;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class FeatureVector implements Serializable {

    public int N;

    // Implement FeatureVector as a sparse vector.
    // Features are stored in a TreeMap which is more efficient to access keySet.
    private TreeMap<Integer, Double> sparseVector;

    // Default constructor.
    public FeatureVector() {
        sparseVector = new TreeMap<>();
        N = 0;
    }

    // Add a new value into vector.
    public void add(int index, double value) {
        // TODO Auto-generated method stub
        // If the value is 0, remove this index from vector.
        // A feature that not exists in the vector means its value is 0.
        if (value == 0.0) {
            sparseVector.remove(index);
        } else {
            sparseVector.put(index, value);
            N++;
        }
    }

    // Get the value with given index.
    public double get(int index) {
        // TODO Auto-generated method stub
        // Return the value by access the TreeMap.
        // If this feature doesn't exist in the vector, return 0.
        if (sparseVector.containsKey(index))
            return sparseVector.get(index);
        else
            return 0.0;
    }

    // Return the entrySet of the TreeMap.
    public Set<Map.Entry<Integer, Double>> entrySet() {
        return sparseVector.entrySet();
    }

    public Set<Integer> getKeySet() {
        return sparseVector.keySet();
    }
}
