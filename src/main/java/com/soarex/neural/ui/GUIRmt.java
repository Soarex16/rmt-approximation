package com.soarex.neural.ui;

import com.bulenkov.darcula.DarculaLaf;
import com.intellij.uiDesigner.core.GridConstraints;
import com.intellij.uiDesigner.core.GridLayoutManager;
import com.intellij.uiDesigner.core.Spacer;
import com.soarex.neural.util.CustomIterationListener;
import com.soarex.neural.util.MouseWheelHandler;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeriesCollection;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.io.*;
import java.util.Random;

import static com.soarex.neural.RegressionMathFuncApproximation.*;

public class GUIRmt extends JFrame {
    private JPanel contentPane;
    private JButton buttonStart;
    private JTextField funcT;
    private JPanel chartJPanel;
    private JTextField fxMin;
    private JTextField fxMax;
    private JTextField fnDataPoints;
    private JProgressBar progressBar;
    private JPanel chartContainer;
    private JTextField layersCfg;
    private JTextField learningRateValue;
    private JButton buttonMoreLess;
    private JPanel optionsMore;
    private JTextField epochsValue;
    private JTextField numOutFunc;
    private JCheckBox randomSeedCheckBox;
    private JTextField seedValue;
    private JTextField batchSizeValue;
    private JSpinner iterationsValue;
    private JComboBox optimisationAlgo;
    private JComboBox lossFunc;
    private JButton buttonLearningGraph;
    private JLabel iterCL;
    private JButton buttonAbout;
    private JButton saveButton;
    private JButton loadButton;
    private JButton buttonStop;
    private JComboBox weightInit;

    private WeightInit wInit = WeightInit.XAVIER;
    private OptimizationAlgorithm optAlg = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
    private LossFunctions.LossFunction lossFn = LossFunctions.LossFunction.MSE;
    private int[] networkCfg;

    private ChartPanel chartPanel;

    private GUILearningProcess learn = new GUILearningProcess();
    private GUIAbout about = new GUIAbout();

    private int totalEpochs = 0;
    private int saveWidth = 1920;
    private int saveHeight = 1080;

    private GUIRmt() {
        try {
            UIManager.setLookAndFeel(new DarculaLaf());
        } catch (UnsupportedLookAndFeelException e) {
            e.printStackTrace();
        }
        $$$setupUI$$$();
        setContentPane(contentPane);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        getRootPane().setDefaultButton(buttonStart);
        try {
            setIconImage(ImageIO.read(getClass().getResourceAsStream("/64.png")));
        } catch (IOException e) {
            e.printStackTrace();
        }

        loadButton.setMnemonic(KeyEvent.VK_L);
        saveButton.setMnemonic(KeyEvent.VK_S);

        iterationsValue.setValue(1);
        optimisationAlgo.setSelectedIndex(4);
        weightInit.setSelectedIndex(4);
        lossFunc.setSelectedIndex(0);

        randomSeedCheckBox.addActionListener(e -> rngOff());
        buttonStart.addActionListener(e -> onStart());
        buttonStop.addActionListener(e -> onStop());
        buttonMoreLess.addActionListener(e -> moreLess());
        buttonAbout.addActionListener(e -> credits());
        buttonLearningGraph.addActionListener(e -> learningGraph());
        saveButton.addActionListener(e -> saveLoadCfg(true));
        loadButton.addActionListener(e -> saveLoadCfg(false));
    }

    public static void main(String[] args) {
        GUIRmt form = new GUIRmt();
        form.setTitle("Math function approximation demo");
        form.pack();
        form.setVisible(true);
    }

    private void onStop() {
        Thread nn = getThreadByName("soarex-neuro");
        if (nn != null) {
            nn.stop();
        }
    }

    private void saveLoadCfg(boolean save) {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select JSON configuration file");
        File cfgFile = null;

        if (save) {
            int choice = chooser.showSaveDialog(this);
            cfgFile = chooser.getSelectedFile();
            if (choice != JFileChooser.APPROVE_OPTION) return;
            JSONObject obj = new JSONObject();

            setVars();

            obj.put("func", func);
            obj.put("xmin", xmin);
            obj.put("xmax", xmax);
            obj.put("nDataPoints", nDataPoints);
            obj.put("nEpochs", nEpochs);
            obj.put("plotFrequency", plotFrequency);
            obj.put("randSeed", randomSeedCheckBox.isSelected());
            obj.put("seed", seed);
            obj.put("iterations", iterations);
            obj.put("batchSize", batchSize);
            obj.put("learningRate", learningRate);
            obj.put("totalEpochs", totalEpochs);
            obj.put("numOutFunc", Integer.parseInt(numOutFunc.getText()));

            JSONArray layersArray = new JSONArray(networkCfg);
            obj.put("networkCfg", layersArray);

            obj.put("weightInit", weightInit.getSelectedIndex());
            obj.put("optAlg", optimisationAlgo.getSelectedIndex());
            obj.put("lossFunc", lossFunc.getSelectedIndex());

            obj.put("saveWidth", saveWidth);
            obj.put("saveHeight", saveHeight);
            try (FileWriter file = new FileWriter(cfgFile)) {
                file.write(obj.toString());
                file.flush();
                file.close();

                if (chartPanel.getChart().getXYPlot().getSeriesCount() > 0 && getThreadByName("soarex-neuro") == null) {
                    ChartUtilities.saveChartAsPNG(new File(cfgFile.getAbsolutePath() + "-approximation.png"), chartPanel.getChart(), saveWidth, saveHeight);
                    ChartUtilities.saveChartAsPNG(new File(cfgFile.getAbsolutePath() + "-learning.png"), learn.chartPanel.getChart(), saveWidth, saveHeight);
                }
            } catch (IOException e) {
                JOptionPane.showMessageDialog(this, "Error while saving configuration!", "Saving error", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        } else {
            int choice = chooser.showOpenDialog(this);
            if (choice != JFileChooser.APPROVE_OPTION) return;
            cfgFile = chooser.getSelectedFile();
            try {
                JSONObject obj = new JSONObject(new JSONTokener(new FileReader(cfgFile)));
                funcT.setText(obj.get("func").toString());
                fxMin.setText(obj.get("xmin").toString());
                fxMax.setText(obj.get("xmax").toString());
                fnDataPoints.setText(obj.get("nDataPoints").toString());
                randomSeedCheckBox.setSelected(obj.getBoolean("randSeed"));
                seedValue.setText(obj.get("seed").toString());
                iterationsValue.setValue(obj.get("iterations"));
                epochsValue.setText(obj.get("nEpochs").toString());
                batchSizeValue.setText(obj.get("batchSize").toString());
                learningRateValue.setText(obj.get("learningRate").toString());
                numOutFunc.setText(obj.get("numOutFunc").toString());

                JSONArray layersArray = obj.getJSONArray("networkCfg");
                StringBuilder bCfg = new StringBuilder();
                networkCfg = new int[layersArray.length()];
                for (int i = 0; i < layersArray.length(); ++i) {
                    networkCfg[i] = layersArray.getInt(i);
                    bCfg.append(networkCfg[i]).append(", ");
                }
                layersCfg.setText(bCfg.substring(0, bCfg.length() - 2));

                wInit = WeightInit.values()[obj.getInt("weightInit")];
                optAlg = OptimizationAlgorithm.values()[obj.getInt("optAlg")];
                lossFn = LossFunctions.LossFunction.values()[obj.getInt("lossFunc")];

                saveWidth = obj.getInt("saveWidth");
                saveHeight = obj.getInt("saveHeight");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    private void setVars() {
        optAlg = OptimizationAlgorithm.values()[optimisationAlgo.getSelectedIndex()];
        lossFn = LossFunctions.LossFunction.values()[lossFunc.getSelectedIndex()];
        wInit = WeightInit.values()[weightInit.getSelectedIndex()];
        func = funcT.getText().toLowerCase();
        xmin = Double.parseDouble(fxMin.getText());
        xmax = Double.parseDouble(fxMax.getText());
        nDataPoints = Integer.parseInt(fnDataPoints.getText());
        batchSize = Integer.parseInt(batchSizeValue.getText());
        nEpochs = Integer.parseInt(epochsValue.getText());
        Number spinnerObj = (Number) iterationsValue.getValue();
        iterations = spinnerObj.intValue();
        learningRate = Double.parseDouble(learningRateValue.getText());
        plotFrequency = nEpochs / Integer.parseInt(numOutFunc.getText());

        int batches = (int) Math.ceil(((double) nDataPoints) / batchSize);
        if (batches < 1) batches = 1;
        totalEpochs = nEpochs * iterations * batches;

        String[] nCfg = layersCfg.getText().replace(" ", "").split(",");
        networkCfg = new int[nCfg.length];
        for (int i = 0; i < nCfg.length; ++i) {
            networkCfg[i] = Integer.parseInt(nCfg[i]);
        }
    }

    private void credits() {
        if (about.isVisible()) {
            about.setVisible(false);
        } else {
            about.setVisible(true);
        }
    }

    private void learningGraph() {
        if (learn.isVisible()) {
            learn.setVisible(false);
        } else {
            learn.setVisible(true);
        }
    }

    private void rngOff() {
        if (randomSeedCheckBox.isSelected()) {
            seedValue.setEnabled(false);
        } else {
            seedValue.setEnabled(true);
        }
    }

    private void moreLess() {
        if (optionsMore.isVisible()) {
            optionsMore.setVisible(false);
            buttonMoreLess.setText("More...");
            this.pack();
        } else {
            optionsMore.setVisible(true);
            buttonMoreLess.setText("Less...");
            this.pack();
        }
    }

    private void onStart() {
        if (getThreadByName("soarex-neuro") != null) {
            JOptionPane.showMessageDialog(this, "Another learning process is already running", "Please, wait", JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        setVars();

        if (randomSeedCheckBox.isSelected()) {
            seed = new Random().nextInt();
            seedValue.setText(String.valueOf(seed));
        } else {
            seed = Integer.parseInt(seedValue.getText());
        }

        final INDArray x = Nd4j.linspace((int) xmin, (int) xmax, nDataPoints).reshape(nDataPoints, 1);
        final INDArray y = getFunctionValues(func, x);
        final DataSetIterator iterator = getTrainingData(x, y, batchSize, rng);

        learn.clearChart();
        learn.setVisible(true);

        Thread networkThread = new Thread(() -> {
            plotFunc(x, y);
            progressBar.setMaximum(totalEpochs);
            progressBar.setValue(0);

            MultiLayerConfiguration config = getParametrisedDeepNetworkConfiguration();
            final MultiLayerNetwork net = new MultiLayerNetwork(config);
            net.init();
            net.setListeners(new CustomIterationListener(1, totalEpochs, iterCL, progressBar, learn));

            final INDArray[] networkPredictions = new INDArray[nEpochs / plotFrequency];

            for (int i = 0; i < nEpochs; ++i) {
                iterator.reset();
                net.fit(iterator);
                if ((i + 1) % plotFrequency == 0) networkPredictions[i / plotFrequency] = net.output(x, false);
            }

            plotFunc(x, y, networkPredictions);
        });
        networkThread.setName("soarex-neuro");
        networkThread.start();
    }

    private Thread getThreadByName(String threadName) {
        for (Thread t : Thread.getAllStackTraces().keySet()) {
            if (t.getName().equals(threadName)) return t;
        }
        return null;
    }

    private MultiLayerConfiguration getParametrisedDeepNetworkConfiguration() {
        NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(optAlg)
                .learningRate(learningRate)
                .weightInit(wInit)
                .updater(Updater.NESTEROVS).momentum(0.75)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(networkCfg[0])
                        .activation("tanh")
                        .build());
        for (int i = 0; i < networkCfg.length; ++i) {
            listBuilder.layer(i + 1, new DenseLayer.Builder().nIn(networkCfg[i]).nOut(i + 1 < networkCfg.length ? networkCfg[i + 1] : networkCfg[i])
                    .activation("tanh")
                    .build());
        }
        listBuilder.layer(networkCfg.length + 1, new OutputLayer.Builder(lossFn)
                .activation("identity")
                .nIn(networkCfg[networkCfg.length - 1]).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
        return listBuilder.build();
    }

    private void plotFunc(final INDArray x, final INDArray y, final INDArray... predicted) {
        XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, x, y, func);

        for (int i = 0; i < predicted.length; ++i) {
            addSeries(dataSet, x, predicted[i], String.valueOf(i));
        }

        JFreeChart chart = ChartFactory.createXYLineChart(
                "",
                "x",
                "y",
                dataSet,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        chart.setBackgroundPaint(UIManager.getColor("window"));
        chart.getXYPlot().setRangePannable(true);
        chart.getXYPlot().setDomainPannable(true);
        chartPanel.setChart(chart);
    }

    private void createUIComponents() {
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "",
                "x",
                "y",
                null,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        chart.setBackgroundPaint(UIManager.getColor("window"));
        chart.getXYPlot().setRangePannable(true);
        chart.getXYPlot().setDomainPannable(true);
        chartPanel = new ChartPanel(chart);
        chartPanel.addMouseWheelListener(new MouseWheelHandler(chartPanel));
        chartPanel.setFillZoomRectangle(false);
        chartJPanel = new JPanel(new BorderLayout());
        chartJPanel.add(chartPanel);
        weightInit = new JComboBox(WeightInit.values());
        optimisationAlgo = new JComboBox(OptimizationAlgorithm.values());
        lossFunc = new JComboBox(LossFunctions.LossFunction.values());
    }

    /**
     * Method generated by IntelliJ IDEA GUI Designer
     * >>> IMPORTANT!! <<<
     * DO NOT edit this method OR call it in your code!
     *
     * @noinspection ALL
     */
    private void $$$setupUI$$$() {
        createUIComponents();
        contentPane = new JPanel();
        contentPane.setLayout(new GridLayoutManager(4, 2, new Insets(10, 10, 10, 10), -1, -1));
        contentPane.setForeground(new Color(-14606047));
        contentPane.setToolTipText("");
        chartContainer = new JPanel();
        chartContainer.setLayout(new GridLayoutManager(1, 1, new Insets(0, 0, 0, 0), -1, -1));
        chartContainer.setForeground(new Color(-4473925));
        contentPane.add(chartContainer, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        chartContainer.add(chartJPanel, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        final JPanel panel1 = new JPanel();
        panel1.setLayout(new GridLayoutManager(2, 8, new Insets(0, 0, 0, 0), -1, -1));
        contentPane.add(panel1, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        progressBar = new JProgressBar();
        panel1.add(progressBar, new GridConstraints(0, 0, 1, 7, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label1 = new JLabel();
        label1.setText("Data points");
        panel1.add(label1, new GridConstraints(1, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label2 = new JLabel();
        label2.setText("Min");
        panel1.add(label2, new GridConstraints(1, 3, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label3 = new JLabel();
        label3.setText("Max");
        panel1.add(label3, new GridConstraints(1, 5, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        fnDataPoints = new JTextField();
        fnDataPoints.setText("");
        fnDataPoints.setToolTipText("Number of data points");
        fnDataPoints.setVisible(true);
        panel1.add(fnDataPoints, new GridConstraints(1, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        fxMin = new JTextField();
        fxMin.setText("");
        fxMin.setVisible(true);
        panel1.add(fxMin, new GridConstraints(1, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        fxMax = new JTextField();
        fxMax.setText("");
        panel1.add(fxMax, new GridConstraints(1, 6, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        iterCL = new JLabel();
        iterCL.setHorizontalAlignment(10);
        iterCL.setText("0/0");
        panel1.add(iterCL, new GridConstraints(0, 7, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JPanel panel2 = new JPanel();
        panel2.setLayout(new GridLayoutManager(1, 3, new Insets(0, 0, 0, 0), -1, -1));
        panel2.setForeground(new Color(-2039584));
        contentPane.add(panel2, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, 1, null, null, null, 0, false));
        final JPanel panel3 = new JPanel();
        panel3.setLayout(new GridLayoutManager(1, 2, new Insets(0, 0, 0, 0), -1, -1));
        panel3.setForeground(new Color(-2039584));
        panel2.add(panel3, new GridConstraints(0, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        buttonStart = new JButton();
        buttonStart.setIcon(new ImageIcon(getClass().getResource("/execute.png")));
        buttonStart.setMargin(new Insets(2, 14, 2, 14));
        buttonStart.setText("Start");
        panel3.add(buttonStart, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonStop = new JButton();
        buttonStop.setIcon(new ImageIcon(getClass().getResource("/cancel.png")));
        buttonStop.setMargin(new Insets(2, 14, 2, 14));
        buttonStop.setText("Stop");
        panel3.add(buttonStop, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        funcT = new JTextField();
        funcT.setText("");
        funcT.setToolTipText("<html>\n<b>Predefined available functions:</b><br>\n<b>abs</b>: absolute value<br>\n<b>acos</b>: arc cosine<br>\n<b>asin</b>: arc sine<br>\n<b>atan</b>: arc tangent<br>\n<b>cbrt</b>: cubic root<br>\n<b>ceil</b>: nearest upper integer<br>\n<b>cos</b>: cosine<br>\n<b>cosh</b>: hyperbolic cosine<br>\n<b>exp</b>: euler's number raised to the power (e^x)<br>\n<b>floor</b>: nearest lower integer<br>\n<b>log</b>: logarithmus naturalis (base e)<br>\n<b>log10</b>: logarithm (base 10)<br>\n<b>log2</b>: logarithm (base 2)<br>\n<b>sin</b>: sine<br>\n<b>sinh</b>: hyperbolic sine<br>\n<b>sqrt</b>: square root<br>\n<b>tan</b>: tangent<br>\n<b>tanh</b>: hyperbolic tangent<br>\n<b>signum</b>: signum function\n</html>");
        panel2.add(funcT, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        final JLabel label4 = new JLabel();
        label4.setText("f(x) =");
        label4.setToolTipText("Function to be approximated");
        panel2.add(label4, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JPanel panel4 = new JPanel();
        panel4.setLayout(new GridLayoutManager(2, 5, new Insets(0, 0, 0, 0), -1, -1));
        panel4.setVisible(true);
        contentPane.add(panel4, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        optionsMore = new JPanel();
        optionsMore.setLayout(new GridLayoutManager(6, 8, new Insets(0, 0, 0, 0), -1, -1));
        optionsMore.setVisible(true);
        panel4.add(optionsMore, new GridConstraints(1, 0, 1, 5, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        layersCfg = new JTextField();
        layersCfg.setText("50, 50, 50, 50, 50, 50, 50, 50");
        layersCfg.setToolTipText("The number of neurons in the hidden layers (separated by commas).");
        layersCfg.setVisible(true);
        optionsMore.add(layersCfg, new GridConstraints(0, 1, 1, 7, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label5 = new JLabel();
        label5.setText("Network configuration");
        optionsMore.add(label5, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        learningRateValue = new JTextField();
        learningRateValue.setText("0.01");
        learningRateValue.setToolTipText("Network learning rate");
        optionsMore.add(learningRateValue, new GridConstraints(1, 1, 1, 4, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label6 = new JLabel();
        label6.setText("Learning rate");
        optionsMore.add(label6, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        numOutFunc = new JTextField();
        numOutFunc.setText("4");
        numOutFunc.setToolTipText("Number of output approximating functions");
        optionsMore.add(numOutFunc, new GridConstraints(2, 1, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label7 = new JLabel();
        label7.setText("Number of output functions");
        label7.setVisible(true);
        optionsMore.add(label7, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        randomSeedCheckBox = new JCheckBox();
        randomSeedCheckBox.setHideActionText(false);
        randomSeedCheckBox.setSelected(true);
        randomSeedCheckBox.setText("random seed");
        optionsMore.add(randomSeedCheckBox, new GridConstraints(3, 7, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label8 = new JLabel();
        label8.setText("Batch size");
        optionsMore.add(label8, new GridConstraints(3, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        batchSizeValue = new JTextField();
        batchSizeValue.setText("100");
        batchSizeValue.setToolTipText("Batch size: i.e., each epoch has nDataPoints/batchSize parameter updates");
        optionsMore.add(batchSizeValue, new GridConstraints(3, 3, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label9 = new JLabel();
        label9.setText("Iterations");
        optionsMore.add(label9, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        iterationsValue = new JSpinner();
        iterationsValue.setToolTipText("Number of iterations per minibatch");
        optionsMore.add(iterationsValue, new GridConstraints(3, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label10 = new JLabel();
        label10.setText("Optimisation algorithm");
        optionsMore.add(label10, new GridConstraints(4, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        optimisationAlgo.setToolTipText("Optimization algorithm to use");
        optionsMore.add(optimisationAlgo, new GridConstraints(4, 1, 1, 7, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label11 = new JLabel();
        label11.setText("Loss function");
        optionsMore.add(label11, new GridConstraints(5, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        lossFunc.setToolTipText("<html><b>Loss function representing the price paid for inaccuracy of predictions</b><br> <b>MSE:</b> Mean Squared Error: Linear Regression<br> <b>EXPLL:</b> Exponential log likelihood: Poisson Regression<br> <b>XENT:</b> Cross Entropy: Binary Classification<br> <b>MCXENT:</b> Multiclass Cross Entropy<br> <b>RMSE_XENT:</b> RMSE Cross Entropy<br> <b>SQUARED_LOSS:</b> Squared Loss<br> <b>NEGATIVELOGLIKELIHOOD:</b> Negative Log Likelihood<br> </html>");
        optionsMore.add(lossFunc, new GridConstraints(5, 1, 1, 7, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        seedValue = new JTextField();
        seedValue.setEnabled(false);
        seedValue.setText("0");
        seedValue.setToolTipText("Random number generator seed, need for reproducibility");
        optionsMore.add(seedValue, new GridConstraints(3, 5, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        epochsValue = new JTextField();
        epochsValue.setText("2000");
        epochsValue.setToolTipText("Number of epochs (full passes of the data)");
        optionsMore.add(epochsValue, new GridConstraints(1, 6, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label12 = new JLabel();
        label12.setText("Epochs");
        optionsMore.add(label12, new GridConstraints(1, 5, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label13 = new JLabel();
        label13.setText("Weight init");
        optionsMore.add(label13, new GridConstraints(2, 3, 1, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        weightInit.setToolTipText("<html>\n<b>Weight initialization scheme</b><br>\n<b>DISTRIBUTION</b>: Sample weights from a provided distribution<br>\n<b>ZERO</b>: Generate weights as zeros<br>\n<b>SIGMOID_UNIFORM</b>: A version of XAVIER_UNIFORM for sigmoid activation functions. U(-r,r) with r=4*sqrt(6/(fanIn + fanOut))<br>\n<b>UNIFORM</b>: Uniform U[-a,a] with a=1/sqrt(fanIn). \"Commonly used heuristic\" as per Glorot and Bengio 2010<br>\n<b>XAVIER</b>: As per Glorot and Bengio 2010: Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)<br>\n<b>XAVIER_UNIFORM</b>: As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))<br>\n<b>XAVIER_FAN_IN</b>: Similar to Xavier, but 1/fanIn -> Caffe originally used this.<br>\n<b>XAVIER_LEGACY</b>: Xavier weight init in DL4J up to 0.6.0. XAVIER should be preferred.<br>\n<b>RELU</b>: He et al. (2015), \"Delving Deep into Rectifiers\". Normal distribution with variance 2.0/nIn<br>\n<b>RELU_UNIFORM</b>: He et al. (2015), \"Delving Deep into Rectifiers\". Uniform distribution U(-s,s) with s = sqrt(6/fanIn)\n</html>");
        optionsMore.add(weightInit, new GridConstraints(2, 5, 1, 3, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JLabel label14 = new JLabel();
        label14.setText("Seed");
        optionsMore.add(label14, new GridConstraints(3, 4, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer1 = new Spacer();
        panel4.add(spacer1, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        buttonMoreLess = new JButton();
        buttonMoreLess.setText("More...");
        panel4.add(buttonMoreLess, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonAbout = new JButton();
        buttonAbout.setText("About");
        panel4.add(buttonAbout, new GridConstraints(0, 4, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonLearningGraph = new JButton();
        buttonLearningGraph.setText("Learning");
        buttonLearningGraph.setToolTipText("Shows a graph of the learning process");
        panel4.add(buttonLearningGraph, new GridConstraints(0, 3, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final JPanel panel5 = new JPanel();
        panel5.setLayout(new GridLayoutManager(1, 2, new Insets(0, 0, 0, 0), -1, -1));
        panel5.setInheritsPopupMenu(true);
        panel5.setOpaque(true);
        panel4.add(panel5, new GridConstraints(0, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, null, null, 0, false));
        loadButton = new JButton();
        loadButton.setIcon(new ImageIcon(getClass().getResource("/open.png")));
        loadButton.setInheritsPopupMenu(true);
        loadButton.setMargin(new Insets(2, 4, 2, 4));
        loadButton.setText("Load");
        loadButton.setToolTipText("Load the configuration of the neural network from file");
        panel5.add(loadButton, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        saveButton = new JButton();
        saveButton.setIcon(new ImageIcon(getClass().getResource("/save.png")));
        saveButton.setMargin(new Insets(2, 4, 2, 4));
        saveButton.setText("Save");
        saveButton.setToolTipText("Save training graphs and approximation and the parameters of the neural network");
        panel5.add(saveButton, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer2 = new Spacer();
        contentPane.add(spacer2, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return contentPane;
    }
}
