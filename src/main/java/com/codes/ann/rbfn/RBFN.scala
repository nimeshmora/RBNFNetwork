import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

import scala.math._

object RBFN{

  val spark = SparkSession.builder()
    .appName("RBFN")
    .master("local")
    .getOrCreate()

  var sc = spark.sparkContext

  def main(args: Array[String]) {

    val data = sc.textFile("data/newdata.csv").map(line=> Vectors.dense(line.split(",").map(_.toDouble)))



    val categories = data.groupBy(r => r(2)).collect()

    val cat1 = sc.parallelize(categories(0)._2.toList)
    val cat2 = sc.parallelize(categories(1)._2.toList)


    var clusters1 = KMeans.train(cat1, 10, 100)
    var clusters2 = KMeans.train(cat2, 10, 100)

    var size:Int = clusters1.clusterCenters.size

    val cells:Array[Double] = new Array[Double](size*cat1.collect().size);


    val sigmas = getSigmas(clusters1, cat1)

    for(i<-sigmas){
      println(i)
    }

    var i:Int = 0

    for(x <- clusters1.clusterCenters){

      for(xn <- cat1.collect()){

        val dist = distance(x.toArray, xn.toArray)
        val phi = getPhi(dist, 1)

        cells(i) = phi

        i += 1
      }

    }
//
//
//
//    var sigmaMatrix = Matrices.dense(size, cat1.collect().size, cells)
//
//    println(sigmaMatrix)



    spark.stop()

  }


  def distance(v1:Array[Double], v2:Array[Double]): Double ={

    var tot:Double = 0.0

    for(i <- 0 until v1.size){
      tot += math.pow(v1(i)-v2(i),2)
    }

    return math.sqrt(tot)
  }


  def getPhi(dist:Double, sigma:Double): Double ={
    return math.exp(- math.pow(dist, 2)/ (2*math.pow(sigma, 2)))
  }

  def getSigmas(model:KMeansModel, categoryData: RDD[org.apache.spark.mllib.linalg.Vector]): Array[Double] ={

    val clusterLabels = model.predict(categoryData)
    val catDataWithclusterLabels = categoryData.zip(clusterLabels)

    val groups = catDataWithclusterLabels.groupBy(r=>r._2)

    var i = 0
    val sigmas:Array[Double] = new Array[Double](groups.count().toInt);
    groups.foreach(args => {


      var totDist:Double = 0.0

      val centroid = model.clusterCenters(i)



      for(a <- args._2){

        println(centroid)
        println(a._1)
        totDist += distance(centroid.toArray, a._1.toArray)
      }

      val avg = totDist/args._2.size
      sigmas(i) = avg
      i += 1
    })

    return sigmas
  }
}
