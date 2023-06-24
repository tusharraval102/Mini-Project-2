public class App {
    public static void main(String[] args) throws Exception {

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        int result = neuralNetwork.predict(0, 0);
        System.out.println(result);

    }
}