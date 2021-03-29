import scalation.analytics.{ExampleAutoMPG, ExampleConcrete, NeuralNet_3L,
  NeuralNet_XL, Optimizer, Perceptron, TranRegression}
import scalation.columnar_db.Relation

object main extends App{
  //AutoMPG dataset
  var x = ExampleAutoMPG.x
  var y = ExampleAutoMPG.y
  var tran = new TranRegression(x,y)
  var perceptron = Perceptron(ExampleAutoMPG.xy)
  val hp = Optimizer.hp
  var net_3L =  NeuralNet_3L(ExampleAutoMPG.xy)
  var net_XL =  NeuralNet_XL(ExampleAutoMPG.xy)
  var t = new p1(x,y, tran)
  var p = new p1(x, y, perceptron)
  var n3L = new p2(x, y, net_3L)
  var nXL = new p2(x, y, net_XL)
  //t.run()
  //p.run()
  //n3L.run()
  //nXL.run()

  //Concrete slump dataset
  val xy = ExampleConcrete.xy
  x = xy.sliceCol(0, xy.range2.length-1)
  y = ExampleConcrete.xy.col(xy.range2.length-1)
  tran = new TranRegression(x,y)
  perceptron = Perceptron(xy)
  net_3L =  NeuralNet_3L(xy)
  net_XL =  NeuralNet_XL(xy)
  t = new p1(x,y, tran)
  p = new p1(x, y, perceptron)
  n3L = new p2(x, y, net_3L)
  nXL = new p2(x, y, net_XL)
  //t.run()
  //p.run()
  //n3L.run()
  //nXL.run()

  //Water toxicity dataset
  val toxicity = Relation("qsar_aquatic_toxicity.csv", "toxicity", null, -1)
  y = toxicity.toVectorD(8)
  x = toxicity.toMatriD(0 to 7)
  tran = new TranRegression(x,y)
  perceptron = Perceptron(toxicity.toMatriD(0 to 8))
  t = new p1(x,y, tran)
  p = new p1(x, y, perceptron)
  net_3L =  NeuralNet_3L(toxicity.toMatriD(0 to 8))
  net_XL =  NeuralNet_XL(toxicity.toMatriD(0 to 8))
  n3L = new p2(x, y, net_3L)
  nXL = new p2(x, y, net_XL)
  //t.run()
  //p.run()
  //n3L.run()
  //nXL.run()

  //Wine dataset
  val wine_quality = Relation("wine.csv", "wine", null, -1)
  y = wine_quality.toVectorD(11)
  x = wine_quality.toMatriD(0 to 10)
  tran = new TranRegression(x,y)
  perceptron = Perceptron(wine_quality.toMatriD(0 to 11))
  t = new p1(x,y, tran)
  p = new p1(x, y, perceptron)
  net_3L =  NeuralNet_3L(wine_quality.toMatriD(0 to 11))
  net_XL =  NeuralNet_XL(wine_quality.toMatriD(0 to 11))
  n3L = new p2(x, y, net_3L)
  nXL = new p2(x, y, net_XL)
  //t.run()
  //p.run()
  //n3L.run()
  //nXL.run()

  //Concrete compression dataset
  val compression = Relation("compress.csv", "compression", null, -1)
  y = compression.toVectorD(8)
  x = compression.toMatriD(0 to 7)
  tran = new TranRegression(x,y)
  perceptron = Perceptron(compression.toMatriD(0 to 8))
  t = new p1(x,y, tran)
  p = new p1(x, y, perceptron)
  net_3L =  NeuralNet_3L(compression.toMatriD(0 to 8))
  net_XL =  NeuralNet_XL(compression.toMatriD(0 to 8))
  n3L = new p2(x, y, net_3L)
  nXL = new p2(x, y, net_XL)
  //t.run()
  //p.run()
  //n3L.run()
  //nXL.run()
}
