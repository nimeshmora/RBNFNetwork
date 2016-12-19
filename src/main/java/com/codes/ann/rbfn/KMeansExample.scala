package com.codes.ann.rbfn

import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors


object KMeansExample {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("KMeansExample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("data/means.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

   
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

   
    clusters.save(sc, "data/output/KMeansModel")
    val sameModel = KMeansModel.load(sc, "data/output/KMeansModel")


    for(x <- clusters.clusterCenters){
      println(x)
    }

    

    sc.stop()
  }
}
// scalastyle:on println
