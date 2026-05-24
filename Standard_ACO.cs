using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

public class AntColonyOptimization
{
    private int _numAnts;
    private int _numIterations;
    private double _evaporationRate;
    private double _initialEvaporationRate;  
    private double _finalEvaporationRate;    
    private double _currentEvaporationRate; 
    private bool _useAdaptiveEvaporation;
    private double _alpha;
    private double _beta;  
    private double _q;  
    private double[][] _pheromone;
    private double[][] _distance;
    private int _numCities;
    private Random _random;
    private double _bestDistance;
    private int[] _bestTour;
    private int _convergencePatience;
    private double _convergenceTolerance;
    private int _iterationsWithoutImprovement;
    private List<ConvergenceRecord> _convergenceHistory;

    public int ConvergenceIteration { get; private set; } = -1;
    public bool HasConverged { get; private set; }
    public List<ConvergenceRecord> ConvergenceHistory => _convergenceHistory;

    public AntColonyOptimization(double[][] distanceMatrix, Random sharedRandom, int numAnts = 30, 
        int numIterations = 100, double evaporationRate = 0.1, 
        double alpha = 1.0, double beta = 2.0, double q = 100.0,
        int convergencePatience = 15, double convergenceTolerance = 1e-6,
        bool useAdaptiveEvaporation = false, double initialEvaporation = 0.10, double finalEvaporation = 0.02)
    {
        _distance = distanceMatrix;
        _numCities = distanceMatrix.Length;
        _numAnts = Math.Min(numAnts, _numCities);
        _numIterations = numIterations;
        _evaporationRate = evaporationRate;
        _useAdaptiveEvaporation = useAdaptiveEvaporation;
        _initialEvaporationRate = initialEvaporation;  
        _finalEvaporationRate = finalEvaporation;      
        _currentEvaporationRate = initialEvaporation;
        _alpha = alpha;
        _beta = beta;
        _q = q;
        _random = sharedRandom; 
        _bestDistance = double.MaxValue;
        _bestTour = new int[_numCities];
        _convergencePatience = Math.Max(1, convergencePatience);
        _convergenceTolerance = convergenceTolerance;
        _iterationsWithoutImprovement = 0;
        _convergenceHistory = new List<ConvergenceRecord>();
        
        _pheromone = new double[_numCities][];
        double initialPheromone = 1.0 / _numCities;
        for (int i = 0; i < _numCities; i++)
        {
            _pheromone[i] = new double[_numCities];
            for (int j = 0; j < _numCities; j++)
            {
                _pheromone[i][j] = initialPheromone;
            }
        }
    }

    public ExperimentResult Solve()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        long initialMemoryBytes = GC.GetTotalMemory(true);
        long peakMemoryBytes = initialMemoryBytes;
        
        var stopwatch = Stopwatch.StartNew();
        
        for (int iteration = 0; iteration < _numIterations; iteration++)
        {
            if (_useAdaptiveEvaporation)
            {
                double progress = (double)iteration / _numIterations;
                _currentEvaporationRate = _initialEvaporationRate - 
                    (_initialEvaporationRate - _finalEvaporationRate) * progress;
            }
            else
            {
                _currentEvaporationRate = _evaporationRate;
            }

            double previousBestDistance = _bestDistance;

            Ant[] ants = new Ant[_numAnts];
            Ant iterationBestAnt = null;
            for (int i = 0; i < _numAnts; i++)
            {
                ants[i] = new Ant(_numCities, _random);
                ants[i].ConstructTour(_distance, _pheromone, _alpha, _beta);
                double tourDistance = CalculateTourDistance(ants[i].Tour, _distance);
                ants[i].TourDistance = tourDistance;

                if (iterationBestAnt == null || tourDistance < iterationBestAnt.TourDistance)
                {
                    iterationBestAnt = ants[i];
                }

                if (tourDistance < _bestDistance)
                {
                    _bestDistance = tourDistance;
                    Array.Copy(ants[i].Tour, _bestTour, _numCities);
                }
            }

            EvaporatePheromone();

            if (iterationBestAnt != null)
            {
                UpdatePheromoneTrail(iterationBestAnt.Tour, iterationBestAnt.TourDistance);
            }

            double improvement = previousBestDistance - _bestDistance;

            if (improvement > _convergenceTolerance)
            {
                _iterationsWithoutImprovement = 0;
            }
            else
            {
                _iterationsWithoutImprovement++;
            }

            if (!HasConverged && _iterationsWithoutImprovement >= _convergencePatience)
            {
                HasConverged = true;
                ConvergenceIteration = iteration + 1;
            }

            _convergenceHistory.Add(new ConvergenceRecord
            {
                Iteration = iteration + 1,
                BestDistance = _bestDistance,
                NoImprovementStreak = _iterationsWithoutImprovement
            });

            Console.WriteLine($"Iteration {iteration + 1}: Best Distance = {_bestDistance:F2}, No-Improvement Streak = {_iterationsWithoutImprovement}");

            if (HasConverged)
            {
                Console.WriteLine($"Convergence detected at iteration {ConvergenceIteration} after {_iterationsWithoutImprovement} stagnant iterations.");
                break;
            }
            
            long currentMemoryBytes = GC.GetTotalMemory(false);
            if (currentMemoryBytes > peakMemoryBytes)
            {
                peakMemoryBytes = currentMemoryBytes;
            }
        }

        if (!HasConverged)
        {
            ConvergenceIteration = _numIterations;
        }

        stopwatch.Stop();
        
        double peakMemoryMb = (peakMemoryBytes - initialMemoryBytes) / (1024.0 * 1024.0);
        
        Console.WriteLine($"Final convergence status: {(HasConverged ? "converged" : "not converged")}; tracked through iteration {ConvergenceIteration}.");

        return new ExperimentResult
        {
            AlgorithmName = "Ant Colony Optimization (Elitist)",
            NumberOfCities = _numCities,
            BestDistance = _bestDistance,
            ConvergenceIteration = ConvergenceIteration,
            RuntimeMs = stopwatch.ElapsedMilliseconds,
            PeakMemoryMb = peakMemoryMb,
            ConvergenceHistory = _convergenceHistory,
            BestTour = _bestTour
        };
    }


    private double CalculateTourDistance(int[] tour, double[][] distances)
    {
        double distance = 0;
        for (int i = 0; i < tour.Length; i++)
        {
            int from = tour[i];
            int to = tour[(i + 1) % tour.Length];
            distance += distances[from][to];
        }
        return distance;
    }

    private void EvaporatePheromone()
    {
        for (int i = 0; i < _numCities; i++)
        {
            for (int j = 0; j < _numCities; j++)
            {
                _pheromone[i][j] *= (1 - _currentEvaporationRate);
            }
        }
    }
    private void UpdatePheromoneTrail(int[] tour, double tourDistance)
    {
        double pheromoneDeposit = _q / tourDistance;
        for (int i = 0; i < tour.Length; i++)
        {
            int from = tour[i];
            int to = tour[(i + 1) % tour.Length];
            _pheromone[from][to] += pheromoneDeposit;
            _pheromone[to][from] += pheromoneDeposit;
        }
    }
}

public class ConvergenceRecord
{
    public int Iteration { get; set; }
    public double BestDistance { get; set; }
    public int NoImprovementStreak { get; set; }
}


public class ExperimentResult
{
    public string AlgorithmName { get; set; }

    public int NumberOfCities { get; set; }

    public double BestDistance { get; set; }

    public int ConvergenceIteration { get; set; }

    public long RuntimeMs { get; set; }

    public double PeakMemoryMb { get; set; }

    public List<ConvergenceRecord> ConvergenceHistory { get; set; }

    public int[] BestTour { get; set; }
    public long MemoryUsageKb { get; set; }
}

public class Ant
{
    public int[] Tour { get; set; }
    public double TourDistance { get; set; }
    private bool[] _visited;
    private Random _random;

    public Ant(int numCities, Random random)
    {
        Tour = new int[numCities];
        _visited = new bool[numCities];
        _random = random;
    }

    public void ConstructTour(double[][] distance, double[][] pheromone, 
        double alpha, double beta)
    {
        int numCities = Tour.Length;
        Array.Clear(_visited, 0, _visited.Length);

        int currentCity = _random.Next(numCities);
        Tour[0] = currentCity;
        _visited[currentCity] = true;


        for (int step = 1; step < numCities; step++)
        {
            int nextCity = SelectNextCity(currentCity, distance, pheromone, alpha, beta);
            Tour[step] = nextCity;
            _visited[nextCity] = true;
            currentCity = nextCity;
        }
    }
    private int SelectNextCity(int currentCity, double[][] distance, 
        double[][] pheromone, double alpha, double beta)
    {
        int numCities = Tour.Length;
        double[] probabilities = new double[numCities];
        double sum = 0;


        for (int i = 0; i < numCities; i++)
        {
            if (!_visited[i])
            {
                double pheromoneLevel = Math.Pow(pheromone[currentCity][i], alpha);
                double desirability = Math.Pow(1.0 / distance[currentCity][i], beta);
                probabilities[i] = pheromoneLevel * desirability;
                sum += probabilities[i];
            }
        }

        if (sum > 0)
        {
            for (int i = 0; i < numCities; i++)
            {
                probabilities[i] /= sum;
            }
        }


        double rand = _random.NextDouble();
        double cumulativeProbability = 0;

        for (int i = 0; i < numCities; i++)
        {
            if (!_visited[i])
            {
                cumulativeProbability += probabilities[i];
                if (rand <= cumulativeProbability)
                {
                    return i;
                }
            }
        }

        for (int i = 0; i < numCities; i++)
        {
            if (!_visited[i])
            {
                return i;
            }
        }

        return 0;
    }
}

class Program
{
    static void Main()
    {
        double[][] cityCoordinates = new double[][]
        {
            new double[] { 0, 0 },
            new double[] { 2, 6 },
            new double[] { 4, 1 },
            new double[] { 6, 7 },
            new double[] { 8, 2 },
            new double[] { 10, 8 },
            new double[] { 12, 3 },
            new double[] { 14, 9 },
            new double[] { 16, 4 },
            new double[] { 18, 10 },
            new double[] { 1, 12 },
            new double[] { 3, 15 },
            new double[] { 5, 11 },
            new double[] { 7, 14 },
            new double[] { 9, 13 },
            new double[] { 11, 16 },
            new double[] { 13, 12 },
            new double[] { 15, 17 },
            new double[] { 17, 11 },
            new double[] { 19, 15 },
            new double[] { 20, 5 },
            new double[] { 22, 12 },
            new double[] { 24, 3 },
            new double[] { 26, 14 },
            new double[] { 28, 6 },
            new double[] { 30, 15 },
            new double[] { 32, 7 },
            new double[] { 34, 16 },
            new double[] { 36, 8 },
            new double[] { 38, 17 },
            new double[] { 21, 20 },
            new double[] { 23, 23 },
            new double[] { 25, 19 },
            new double[] { 27, 22 },
            new double[] { 29, 21 },
            new double[] { 31, 24 },
            new double[] { 33, 20 },
            new double[] { 35, 25 },
            new double[] { 37, 19 },
            new double[] { 39, 23 },
            new double[] { 40, 10 },
            new double[] { 42, 18 },
            new double[] { 44, 9 },
            new double[] { 46, 20 },
            new double[] { 48, 11 },
            new double[] { 50, 19 },
            new double[] { 52, 12 },
            new double[] { 54, 21 },
            new double[] { 56, 13 },
            new double[] { 58, 22 }
        };

        double[][] distanceMatrix = CreateEuclideanDistanceMatrix(cityCoordinates);

        Console.WriteLine("=== Ant Colony Optimization for Traveling Salesman Problem ===");
        Console.WriteLine($"Configuration: 100 Cities, 10 Ants, 100 Iterations\n");

        int runs = 30;
        var allResults = new List<ExperimentResult>();
        Random sharedRandom = new Random(42); 

        for (int run = 1; run <= runs; run++)
        {
            Console.WriteLine($"\n--- Run {run}/{runs} ---");

            var aco = new AntColonyOptimization(
                distanceMatrix,
                sharedRandom,
                numAnts: 10,
                numIterations: 100,
                evaporationRate: 0.05,
                alpha: 0.5,
                beta: 3.0,
                q: 100.0,
                convergencePatience: 20,
                useAdaptiveEvaporation: false
            );

            var result = aco.Solve();
            allResults.Add(result);

            Directory.CreateDirectory("ACO-Standard");
            string perRunCsv = $"ACO-Standard/experiment_run_{run:00}_history.csv";
            var perSb = new StringBuilder();
            perSb.AppendLine("Iteration,BestDistance,NoImprovementStreak");
            foreach (var rec in result.ConvergenceHistory)
            {
                perSb.AppendLine($"{rec.Iteration},{rec.BestDistance:F6},{rec.NoImprovementStreak}");
            }
            File.WriteAllText(perRunCsv, perSb.ToString());
            Console.WriteLine($"Wrote per-run CSV: {Path.GetFullPath(perRunCsv)}");
        }

        Directory.CreateDirectory("ACO-Standard");
        string csvPath = "ACO-Standard/experiment_results.csv";
        var sb = new StringBuilder();
        sb.AppendLine("Run,AlgorithmName,NumberOfCities,BestDistance,ConvergenceIteration,RuntimeMs,PeakMemoryMb");
        for (int i = 0; i < allResults.Count; i++)
        {
            var r = allResults[i];
            sb.AppendLine($"{i + 1},{r.AlgorithmName},{r.NumberOfCities},{r.BestDistance:F6},{r.ConvergenceIteration},{r.RuntimeMs},{r.PeakMemoryMb:F3}");
        }
        File.WriteAllText(csvPath, sb.ToString());
        Console.WriteLine($"\nWrote CSV: {Path.GetFullPath(csvPath)}");

        var last = allResults.Last();
        Console.WriteLine("\n=== Last Run Summary ===");
        Console.WriteLine($"Algorithm: {last.AlgorithmName}");
        Console.WriteLine($"Best Distance: {last.BestDistance:F2}");
        Console.WriteLine($"Convergence Iteration: {last.ConvergenceIteration}");
        Console.WriteLine($"Runtime: {last.RuntimeMs} ms");
        Console.WriteLine($"Peak Memory Usage: {last.PeakMemoryMb:F2} MB");
    }

    static void PrintMatrix(double[][] matrix)
    {
        for (int i = 0; i < matrix.Length; i++)
        {
            for (int j = 0; j < matrix[i].Length; j++)
            {
                Console.Write($"{matrix[i][j],6:F2} ");
            }
            Console.WriteLine();
        }
    }

    static double[][] CreateEuclideanDistanceMatrix(double[][] coordinates)
    {
        int numCities = coordinates.Length;
        double[][] matrix = new double[numCities][];

        for (int i = 0; i < numCities; i++)
        {
            matrix[i] = new double[numCities];
            for (int j = 0; j < numCities; j++)
            {
                if (i == j)
                {
                    matrix[i][j] = 0;
                    continue;
                }

                double deltaX = coordinates[i][0] - coordinates[j][0];
                double deltaY = coordinates[i][1] - coordinates[j][1];
                matrix[i][j] = Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
            }
        }

        return matrix;
    }
}
