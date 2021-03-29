import scalation.analytics.{Fit, Regression, PredictorMat, PredictorMat2}
import scalation.linalgebra.{MatriD, VectoD, VectorD}
import scalation.util.banner
import scalation.plot.Plot
import scalation.analytics.PredictorMat2.{analyze, analyze2}

import scala.collection.mutable.Set

class p2(var x : MatriD, var y : VectoD, var rg : PredictorMat2){

  val qoFs = Map("rSq" -> Fit.index_rSq, "rBarSq" -> Fit.index_rSqBar, "aic" -> Fit.index_aic)
  def eda(){
    //EDA
    for (i <- x.range2){
      new Plot(x.col(i), y)
    }
  }

  def fsel(){
    //Forward Selection and plotting
    for(i <- 0 until qoFs.size) { // looping through the qoF measures
      val qoF_name = qoFs.keys.toArray.array(i)
      val qoF = qoFs(qoF_name)
      var set = Set(0)
      var vec = new VectorD(0)
      banner("forward seleciton: " + qoF_name)
      while(!set.contains(-1)){
        val forward_features = rg.forwardSel(cols = set, index_q = qoF) //forward selection
        set.add(forward_features._1) //keeping track of new predictors
        if(forward_features._1 != -1){
          vec = vec ++ forward_features._2.fitMap(0)(qoF_name).toDouble//logging new qoF metric
        }
      }
      set.remove(-1)
      println(set)
      new Plot(null, vec, lines = true, _title= s"Forward selection: ${qoF_name} v.s. number of predictors")
    }

    var set = Set(0)
    var vec = new VectorD(0)
    while(!set.contains(-1)){
      val forward_features = rg.forwardSel(cols = set, index_q = Fit.index_rSq)
      set.add(forward_features._1)
      if(forward_features._1 != -1) {
        vec = vec ++ forward_features._2.crossValidate(10)(0).mean //rSqCV
      }
    }
    set.remove(-1)
    new Plot(null, vec, lines = true, _title= "Forward selection: rSqCV v.s. number of predictors")
  }

  def belim(): Unit ={
    //Backwards Elimination and plotting
    for(i <- 0 until qoFs.size) {
      val qoF_name = qoFs.keys.toArray.array(i)
      val qoF = qoFs(qoF_name)
      var set = Set[Int]()
      for (q <- 0 until x.range2.length){//adding all predictors to the initial set
        set.add(q)
      }
      var vec = new VectorD(0)
      banner("backward elimination: " + qoF_name)
      for(j <- 1 to x.range2.length-1) {
        println(set.size, x.range2.length-1)
        val backward_features = rg.backwardElim(cols = set, index_q = qoF)
        vec = vec ++ backward_features._2.fitMap(0)(qoF_name).toDouble//logging new qoF metric
        set.remove(backward_features._1) //removing the eliminated features
      }
      new Plot(null, vec, lines = true, _title= s"Backward elimination: ${qoF_name} v.s. number of eliminated predictors")
    }

    //Backwards Elimination and plotting for rSqCV
    var vec = new VectorD(0)
    var set = Set[Int]()
    for (q <- 0 until x.range2.length){
      set.add(q)
    }
    for(j <- 1 to x.range2.length-1) {
      val backward_features = rg.backwardElim(cols = set, index_q = Fit.index_rSq)
      vec = vec ++ backward_features._2.crossValidate(10)(0).mean //rSqCV
      set.remove(backward_features._1)
    }
    new Plot(null, vec, lines = true, _title= "Backward elimination: rSqCV v.s. number of predictors")
  }

  def stepRegression(set : Set[Int], qoF: Int, qoF_name : String): Tuple2[Set[Int], Double] ={
    var fSet = set.clone() //the forward set is equal to the current set of features
    val forward_features = rg.forwardSel(cols = fSet, index_q = qoF) //performing a step of forward selection
    fSet.add(forward_features._1) //keeping track of added features
    var bSet = fSet.clone() //backward elimination set is equal to the forward selection set
    val backward_features = rg.backwardElim(cols = bSet, index_q = qoF) //performing one step of elimination
    bSet.remove(backward_features._1) //removing the feature from the backward set
    val bqOf = backward_features._2.fitMap(0)(qoF_name).toDouble //getting qoF value for both forward and backward selection
    val fqOf = forward_features._2.fitMap(0)(qoF_name).toDouble
    if (fqOf > bqOf){ //returning the features and qoF value for the better performing set
      (fSet, fqOf)
    }else{
      (bSet, bqOf)
    }
  }

  def stepRegressionCV(set : Set[Int]): Tuple2[Set[Int], Double] ={
    var fSet = set.clone()
    val forward_features = rg.forwardSel(cols = fSet, index_q = Fit.index_rSq)
    fSet.add(forward_features._1)
    var bSet = fSet.clone()
    val backward_features = rg.backwardElim(cols = bSet, index_q = Fit.index_rSq)
    bSet.remove(backward_features._1)
    val bqOf = backward_features._2.crossValidate(10)(0).mean //qoF value is now the cross validated rSq (taking the mean)
    val fqOf = forward_features._2.crossValidate(10)(0).mean
    if (fqOf > bqOf){
      (fSet, fqOf)
    }else{
      (bSet, bqOf)
    }
  }

  def stepRegressionAll(): Unit = {
    for (i <- 0 until qoFs.size) { //looping over all qoF metrics
      val qoF_name = qoFs.keys.toArray.array(i)
      val qoF = qoFs(qoF_name)
      var mSet = stepRegression(Set(0), qoF, qoF_name) //calling stepwise regression
      var next = true
      var vec = new VectorD(0)
      vec = vec ++ mSet._2
      while (next) {
        if (mSet._1.size < x.range2.length-1) { //only perform a step if there are features left
          var tSet = stepRegression(mSet._1, qoF, qoF_name)
          if (tSet._2 > mSet._2) { //if the step improves the model, take the new features
            mSet = tSet
          } else { //if no improvement is found, stop
            next = false
          }
        } else {
          next = false
        }
        vec = vec ++ mSet._2 //adding qoF metric to a vector for plotting
      }
      new Plot(null, vec, lines = true, _title= s"Stepwise regression: ${qoF_name} v.s. number of predictors")
    }

    var mSet = stepRegressionCV(Set(0))
    var next = true
    var vec = new VectorD(0)
    vec = vec ++ mSet._2
    while (next) {
      if (mSet._1.size < x.range2.length-1) {
        var tSet = stepRegressionCV(mSet._1) //same as above except calling the CV method
        if (tSet._2 > mSet._2){
          mSet = tSet
        } else {
          next = false
        }
      } else {
        next = false
      }
      vec = vec ++ mSet._2
    }
    new Plot(null, vec, lines = true, _title= s"Stepwise regression: rSqCV v.s. number of predictors")
  }

  def run(): Unit ={
    analyze2(rg)
    fsel()
    belim()
    stepRegressionAll()
  }
}



