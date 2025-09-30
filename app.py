from pipeline.prediction_pipeline import Recommender

if __name__ == '__main__':
    model = Recommender(n=15)
    animes = model.recommend()
    print(animes)