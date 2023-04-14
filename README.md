# Рекомендательные Системы в Продакшене
## Решенное дз #1
### Инструкция по запуску эксперимента
```commandline
cd botify
sudo docker compose up -d --build
cd ../sim
export PYTHONPATH=${PYTHONPATH}:.
python sim/run.py --episodes 1000 --config config/env.yml single --recommender remote --seed 31337 
docker cp recommender-container:/app/log/ /tmp/recsys
cd ../botify
docker compose stop
```
**Препроцессинг данных, описание и тренировка модели и отчет об эксперименте** 
находятся в ноутбуке made-recsys-hw.ipynb в папке jupyter.
