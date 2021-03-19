/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pl.sprint.fasttexttest;

import com.mayabot.nlp.fasttext.FastText;
import com.mayabot.nlp.fasttext.ScoreLabelPair;
import com.mayabot.nlp.fasttext.args.InputArgs;
import com.mayabot.nlp.fasttext.loss.LossName;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;
/**
 *
 * @author Sławomir Kostrzewa
 */
public class Main {
    
    private static final float MIN_SCORE = 70;
    
    public static void main(String[] args) throws Exception {
                 
        String modelPath = "D:/Sprint/SRC/sprint-bot-server/models/tak-nie-model";
        String pathToVec = "D:/Sprint/SRC/sprint-bot-server/models/ref_pl.vec";
        
        String type = "test"; //train|test
        
                
        if(args.length > 1)
        {
            type = args[0];
            modelPath = args[1];
            pathToVec = args[2];
            
        }
                                            
        
        if(type.equals("train"))
            trainSupervised(modelPath, null);        
                      
        test(modelPath);

        
    }
    
    private static void test(String model) throws IOException
    {
        BufferedReader obj = new BufferedReader(new InputStreamReader(System.in));   
        String str;   
        
        System.out.println();   
        System.out.println("Enter lines of text.");   
        System.out.println("Enter 'stop' to quit.");          
        do 
        {   
            str = obj.readLine();   
            checkWord(model, str);   
        }while(!str.equals("stop"));   
    }
   
    
    private static void trainSupervised(String modelName, String pathToVec) throws Exception            
    {
        File trainFile = new File(modelName + ".txt");
        InputArgs inputArgs = new InputArgs();
        inputArgs.setMinCount(1);
        inputArgs.setWordNgrams(2);
        inputArgs.setMinn(3);
        inputArgs.setMaxn(5);
        inputArgs.setBucket(2000000);
        inputArgs.setLr(0.2);
        inputArgs.setDim(100);
        inputArgs.setLoss(LossName.softmax);
        inputArgs.setEpoch(30);
        
        if(pathToVec != null && pathToVec.length() > 0)
            inputArgs.setPretrainedVectors(new File(pathToVec));

        FastText model = FastText.trainSupervised(trainFile, inputArgs);
                
        model.saveModel(modelName);
        

    }
            
    
    
    private static void checkWordKlasic(String modelName, String input, String type)
    {
        FastText model = FastText.Companion.loadCppModel(new File(modelName));
        
        List<ScoreLabelPair> result = model.predict(Arrays.asList(input.split(" ")), 2, 0);
        float diff = (result.get(0).getScore()*100) - (result.get(1).getScore() * 100);
        
        
        if(type.equals("L"))
        {
            System.out.println("diff: " + diff);
            System.out.println("result : " + result.get(0).getLabel().toUpperCase() + " score: " + result.get(0).getScore());
            if(diff < 0.2)
                System.out.println("result : " + result.get(1).getLabel().toUpperCase() + " score: " + result.get(1).getScore());    
        
        }
        else
        {
            List<ScoreLabelPair> resultSkipgram = model.nearestNeighbor(input,5);
                                        
            for(ScoreLabelPair r : resultSkipgram)
                System.out.format("Result predicted:%f word: %s \n",r.component1(), r.component2());
        }
        
                                        
    }
    
    
    private static void checkWord(String modelName, String input)
    {
        FastText model = FastText.Companion.loadModel(new File(modelName), true);  
        
        List<ScoreLabelPair> result = model.predict(Arrays.asList(input.split(" ")), 2, 0.0f);
        float diff = (result.get(0).getScore()*100) - (result.get(1).getScore() * 100);
                

        if(result.get(0).getScore()*100 < MIN_SCORE)
        {
            System.out.println("\nNO QUALIFICATION:\t" + input);

            //System.out.println("diff: " + diff);
            System.out.println("result : " + result.get(0).getLabel() + " score: " + result.get(0).getScore());
        }
        else
        {
            System.out.println("\nOK:\t" + input);
            System.out.println("result : " + result.get(0).getLabel() + " score: " + result.get(0).getScore());
        
        }
           
        
        
                                        
    }
                    
    
    
    
    private static void trainSkipgram() throws Exception
    {
        
        File trainFile = new File("news_pl_2020_1-2.txt");
        InputArgs inputArgs = new InputArgs();
        inputArgs.setMinCount(5);
        inputArgs.setMinn(3);
        inputArgs.setMaxn(6);
        inputArgs.setDim(100);
        inputArgs.setEpoch(7);
        FastText model = FastText.trainSkipgram(trainFile, inputArgs);        
        model.saveModel("news_pl_2020_1-2");
        
        //model.test(new File("vectra.test"),1,0,true);
        
        List<ScoreLabelPair> result = model.nearestNeighbor("wirus",5);
                
        for(ScoreLabelPair r : result)
            System.out.format("Result predicted:%f word: %s \n",r.component1(), r.component2());
    }
    
}
