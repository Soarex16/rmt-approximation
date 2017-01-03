package com.soarex.neural.util;

import com.soarex.neural.ui.GUILearningProcess;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;

/**
 * Created by soarex on 26.12.16.
 */
public class CustomIterationListener implements IterationListener {
    private static final Logger log = LoggerFactory.getLogger(CustomIterationListener.class);
    private int printIterations = 10;
    private boolean invoked = false;
    private long iterCount = 0;
    private JProgressBar bar;
    private GUILearningProcess scoreData;
    private JLabel progressInf;
    private String total;


    public CustomIterationListener(int printIterations, int totalIter, JLabel progressLabel, JProgressBar bar, GUILearningProcess learnData) {
        this.printIterations = printIterations;
        this.bar = bar;
        scoreData = learnData;
        progressInf = progressLabel;
        total = "/".concat(String.valueOf(totalIter));
    }

    @Override
    public boolean invoked(){ return invoked; }

    @Override
    public void invoke() { this.invoked = true; }

    @Override
    public void iterationDone(Model model, int iteration) {
        ++iterCount;
        double result = model.score();
        bar.setValue((int) iterCount);
        progressInf.setText(String.valueOf(iterCount).concat(total));
        scoreData.learnSeries.add(iterCount, result);
        scoreData.score.setText(String.valueOf(result));
        scoreData.iter.setText(String.valueOf(iterCount));
        if(printIterations <= 0)
            printIterations = 1;
        if(iterCount % printIterations == 0) {
            invoke();
            log.info("Score at iteration " + iterCount + " is " + result);
        }
    }
}
