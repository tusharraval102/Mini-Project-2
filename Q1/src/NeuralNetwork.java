public class NeuralNetwork {

    private final int bias = -1;
    private final int weight1 = 1;
    private final int weight2 = 1;

    private int activationFunction(int sum) {
        if (sum <= 0)
            return 0;

        return 1;
    }

    public int predict(int input1, int input2) {
        int sum = (input1 * weight1) + (input2 * weight2) + bias;

        return activationFunction(sum);
    }
}
