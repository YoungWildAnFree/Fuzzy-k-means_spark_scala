package com.hbdx.spark.fuzzykmeans

import java.io._
import java.util.Random

import org.apache.spark.{Partitioner, SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

import Array._
/**
  * Created by new on 2017/4/21.
  */
object FuzzyKmeans extends Exception{
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Fuzzy Kmeans")
    val sc = new SparkContext(conf)
    // 第一个参数为K值分几类
    //第二个参数为m值 模糊程度 一般为2
    //第三个为输入路径
    //第四个为输出路径
    val k = args(0).toInt
    val m = args(1).toDouble
    val input = args(2)
    val output = args(3)

    val starttime = System.currentTimeMillis()
    val threshold = 0.1

    var IterationStatu = true

    val rdd = sc.textFile(input)

    // iris.data  3 2 G:\\208计算机\\D盘\\SparkKmeans\\input\\iris.data d:\\fuzzyoutRDDiris
//      val points = rdd.map(x => {
//          val tmp = x.split(",").take(4).map(_.toDouble)
//          Vectors.dense(tmp)
//      }).cache()

    //pokerhand.data 6 2 G:\\208计算机\\D盘\\SparkKmeans\\input\\pokerhand.data d:\\fuzzyoutRDDpoker
        val points = rdd.map(x => {
          val tmp = x.split(",").map(_.toDouble)
          Vectors.dense(tmp)
        }).cache()

    // SUSY 数据集 2 2 G:\\208计算机\\data\\fuzzySUSY\\SUSY.csv d:\\fuzzyoutSUSY
//    val points = rdd.map(x => {
//      val tmp = x.split(",").slice(1,19).map(_.toDouble)
//      Vectors.dense(tmp)
//    }).coalesce(1).cache()


    print("-------------------------------------------------\n")
    print("文件已读入 \n")
    //points.foreach(println)
    val n = points.count() //样本点个数
    println("样本点个数为 " + n)
    print("--------------------------------------------------\n")
     val first = points.first()
     val d= first.size
    println("样本的维度是"+ d)
    print("-------------------------------------------------\n")


    print("初始化隶属度\n") // 初始隶属度每一行为样本点Xi的隶属度 每一行相加等于1
    var memdeg = ofDim[Double](n.toInt,k )

    var tmp =0.0
    for (i <- 0 until n.toInt) {
      for (j <- 0 until k) {
        memdeg(i)(j)=math.random
        tmp = memdeg(i)(j) + tmp
      }
      for(q<- 0 until k){
        memdeg(i)(q) = memdeg(i)(q)/tmp
        //print(memdeg(i)(q)+"  ")
      }
      tmp =0.0
      //println()
    }

    print("初始隶属度完毕")
    var memdegRDD = sc.parallelize(memdeg,1)

    val centers = ofDim[Double](k,d.toInt)
    println(memdegRDD.first().size + "   "+memdegRDD.count().toString)
    println(memdegRDD.first().toVector)

    def mcifang(x:Array[Double],m:Double): Array[Double] ={
      var tmp = new Array[Double](x.length)
      for(i<- 0 until x.length){
       tmp(i) =math.pow(x(i),m)
      }
      tmp
    }

    def meiliehe(x: Array[Array[Double]], y: Array[Array[Double]]): Array[Array[Double]] = {
      var t = new Array[Array[Double]](k)

      for (i <- 0 until k) {
        var tmp = new Array[Double](d)
        for (j <- 0 until d) {
          tmp(j) = x(i)(j) + y(i)(j)
        }
        t(i) = tmp
      }
      t
    }
    //隶属度每列之和
    def arrayadd(x: Array[Double], y: Array[Double]): Array[Double] = {
      var t = new Array[Double](k)
      for (i <- 0 until k) {
        t(i) = x(i) + y(i)
      }
      t
    }

    var IterTimes= 0
    val diedaistart = System.currentTimeMillis()
    while (IterationStatu) {
      IterTimes += 1
      //println("第" + IterTimes + "次 迭代")

      //println("memdegRDD"+memdegRDD.first().toVector)
      var memdegMRDD = memdegRDD.map(x => {
        mcifang(x, m)
      }).cache()

      println("memdegMRDD 的数量" + memdegMRDD.count())
      //println("memdegMRDD"+memdegMRDD.first().toVector)
      //  println("----------------------------------------------")
      //Uij的m次方乘以 Xi   ([],[])  -> (()()())Array[Array[Double](d)](k)
      var pointswithmemdegMRDD = points.zip(memdegMRDD).map(x => {
        var t = new Array[Array[Double]](k)
        //var tmp = new Array[Double](d)
        for (i <- 0 until k) {
          var tmp = new Array[Double](d)
          for (j <- 0 until d) {

            tmp(j) = x._2(i) * x._1(j)
          }
          t(i) = tmp
        }
        t
      }).repartition(4).cache()

      //Cij的分子   (()()())Array[Array[Double](d)](k)
      var CijFenzi = pointswithmemdegMRDD.reduce((x, y) => meiliehe(x, y))
    //  println("CijfenziRDD"+CijFenzi(1).toVector)

     // println("pointswithmemdegMRDD"+pointswithmemdegMRDD.first()(1).toVector)

      var CijFenmu = memdegMRDD.reduce((x, y) => arrayadd(x, y))


     // println("Cijfenmu"+CijFenmu.mkString("  "))

      var qiancenters = Array.ofDim[Double](k,d)
      for (i <- 0 until k) {
        for (j <- 0 until d) {
           qiancenters(i)(j) = CijFenzi(i)(j) / CijFenmu(i)
        }
      }


      var uijRDD = points.map(x => {
        var tmp = new Array[Double](k)
        for (i <- 0 until k) {
          tmp(i) = 1 / Uij(x, qiancenters, i, m)
        }
        tmp
      })

     // println("uijRDD."+uijRDD.first().toVector)
      //比较UijRDD 与memdegRDD 其中变化最大的与阈值比较 最大差值 小于 阈值的跳出循环
      var bijiao = 0.0

      println("--------------------------")
      var converge = memdegRDD.zip(uijRDD).map(x => {
        for (i <- 0 until k) {
          var tmp = 0.0
          tmp = x._1(i) - x._2(i)
          if (bijiao < tmp) {
            bijiao = tmp
          }
        }
        bijiao
      }).max()
//      // 根据centers前后变化 判断阈值
//      var converge = 0.0
//      for (i<- 0 until k){
//        for (j<- 0 until d){
//          var tmp=0.0
//          tmp=qiancenters(i)(j)- centers(i)(j)
//          if(converge<tmp){converge=tmp}
//        }
//      }

     // println("bijiaozhi :" + converge)
      //converge < threshold &&
      if (IterTimes==10) {
        println("退出循环" + "迭代次数"+IterTimes)
        IterationStatu = false

      }
      for (i<- 0 until k){
        for (j<- 0 until d){
          centers(i)(j)= qiancenters(i)(j)
        }
      }
      memdegRDD = uijRDD
      pointswithmemdegMRDD.unpersist()
      memdegMRDD.unpersist()

    }
    val diedaitime  = System.currentTimeMillis()-diedaistart
    var centersRDD = sc.parallelize(centers)
    centersRDD.map(Vectors.dense(_)).saveAsTextFile(output+"\\centers")
    memdegRDD.map(Vectors.dense(_)).saveAsTextFile(output+"\\memdeg")
    val stoptime = System.currentTimeMillis()-starttime
    println("整个程序运行" + stoptime + "\t" + "迭代运行时间" + diedaitime +"\t" + "迭代次数"+IterTimes)




    //println(menmdegMRDD.first() + "   "+menmdegMRDD.count().toString)

    //println("准备好进行迭代 " + IterationStatu)
//    while (false) {
//      IterTimes += 1
//      println("第" + IterTimes + "次 迭代")
//      println("Uij 的M次方的矩阵 ")
//      var MemdegM = ofDim[Double](n.toInt, k) //表示Uij的m次方的矩阵
//      for (i <- 0 until n.toInt) {
//        for (j <- 0 until k) {
//          MemdegM(i)(j) = math.pow(memdeg(i)(j), m)
//          //print(MemdegM(i)(j) + "  ")
//        }
//       // println()
//      }
//      //memdegMRDD 是Uij隶属度矩阵的m次方
//      val memdegMRDD=memdegRDD.map(x=>{
//        Vectors.dense(mcifang(x,m))
//      })
//
//      val Memdegmbd = sc.broadcast(MemdegM)
//      println("----------------------------------------------")
//
//      val pointswithmemdegMRDD = points.union(memdegMRDD)
//      pointswithmemdegMRDD.first().size
//      val acc = sc.longAccumulator
//
//      var fenzi = new Array[RDD[Array[Double]]](k)
//      for (i <- 0 until k) {
//        fenzi(i) = points.map(x => {
//          val tmp = updataCenters(x, Memdegmbd.value, acc.value.toInt, i)
//          acc.add(1L)
//          tmp
//        })
//        acc.reset()
//        //println(fenzi(i).count() + "   " + acc.value)
//
//      }
//      var zonghe = new Array[Array[Double]](k)
//      for (i <- 0 until k) {
//        var arr = new Array[Double](d)
//        zonghe(i) = fenzi(i).reduce(VectorAdd(_, _, arr, d))  //求和
//        //求出Uij乘以Xi的总和  当前j为0
//        //下面求Uij矩阵第O列的总和
//        //zonghe(i).toVector.foreach(x => print(x + "  "))
//      }
//      println("\n-----------------------------------------------\n")
//      println("隶属度每列和:")
//      var SumJ = new Array[Double](k)// Cj分母 Uij的m次方 n个样本求和
//      for (j <- 0 until k) {
//        for (i <- 0 until (n.toInt)) {
//          SumJ(j) = Memdegmbd.value(i)(j) + SumJ(j)
//
//        }
//       // print(SumJ(j) + "   ")
//      }
//      println("\n````````````````````````````````\n")
//      println("K个中心点：")
//      //求出中心点
//      for (i <- 0 until k) {
//        for (j <- 0 until d) {
//          centers(i)(j) = zonghe(i)(j) / SumJ(i)
//         // print(centers(i)(j) + "    ")
//        }
//        //println()
//      }
//      Memdegmbd.unpersist()
//      println("Cij 已经确定，之后根据确定的Cij更新Uij的值")
//      println("计算Uij")
//      acc.reset()
//      println("\n-----------------------------------------\n")
//      var uij = new Array[RDD[Double]](k)
//
//      for (i <- 0 until k) {
//        uij(i)= points.map(x => {
//          var result = 1 / Uij(x, centers, i, m)
//          result
//        })
//      }
//      var arrra = new Array[Array[Double]](k) //接收uij RDD中的值
//        for(j<- 0 until k) {
//          var c =uij(j).count().toInt
//          arrra(j)= uij(j).take(c)
//        }
//      // 更新隶属度矩阵前 进行判断 计算 arrra得到的隶属度 与之前的memdeg矩阵中的隶属度
//      //差值中最大是否小于给定的阈值  小于则 将arrra赋值给memdeg 然后退出循环
//        var a =0.0
//        for (i<- 0 until n.toInt){
//          for(j<- 0 until k){
//            var b=0.0
//            b =arrra(j)(i) - memdeg(i)(j)
//            if(a< b){a=b}
//          }
//        }
//
//
//     // println(a+"比较----------------------------")
//      if(a<threshold){IterationStatu=false}
//
//      for(i<- 0 until n.toInt){  //将从uij中取出的值赋值给memdeg 隶属度矩阵
//        for(j<- 0 until k){
//          memdeg(i)(j) = arrra(j)(i)
//        }
//      }
////      println("隶属度更新为：")
////      for(i<- 0 until n.toInt){
////        for (j<- 0 until k){
////          print(memdeg(i)(j)+"  ")
////        }
////        println()
////      }
//      println("---------------------------------")
//    }
    // 迭代完成后 产生K个中心点 和Uij 隶属度矩阵
    //根据centers和memdeg 对数据中个点进行聚类
    //points.map(x=> (x, ))
//    val outputfile = new File(output)
//    outputfile.mkdirs()
//    val memdegout = new PrintWriter(new File(output.concat("\\memdeg.txt")))
//    for(i<- 0 until n.toInt){
//      for (j <- 0 until k){
//          memdegout.write(memdeg(i)(j).toString+ "\t")
//      }
//      memdegout.write("\r\n")
//    }
//    memdegout.close()
//    val centerout = new PrintWriter(new File(output.concat("\\center.txt")))
//    for(i<- 0 until k){
//      for(j<- 0 until d) {
//        centerout.write(centers(i)(j).toString + "\t")
//      }
//        centerout.write("\r\n")
//    }
//    centerout.close()
  }

  // Uij的m次方*Xi
  def   updataCenters(x:Vector,u:Array[Array[Double]],i:Int,j:Int)={
    val d= x.size
    var tmp = new Array[Double](d)
    for (q<- 0 until d){
      tmp(q) = u(i)(j) * x(q)
    }
    tmp
  }


//Cj 公式分子 求和
  def VectorAdd(x:Array[Double],y:Array[Double],arr:Array[Double],d:Int)={
    for (i<- 0 until d){
      arr(i)= x(i)+y(i)
    }
    arr
  }
  def EuclideanDistance(x:Vector,y:Array[Double]):Double={
    var distance:Double =0.0
    for (i <- 0 until y.length){
      distance = math.pow(x(i)-y(i),2)+distance
    }
    distance= math.sqrt(distance)
    distance
  }
//Uij公式  样本与所得各个中心点的距离 的2/(m-1)次方 求和
  def Uij(x:Vector,y:Array[Array[Double]], j:Int,m:Double): Double ={
    var sum= 0.0
    val k = y.length
    var fenzi = EuclideanDistance(x,y(j))
    for(i<- 0 until k){
      sum=math.pow((fenzi/(EuclideanDistance(x,y(i)))),2/(m-1))+sum
    }
      sum
  }


}
