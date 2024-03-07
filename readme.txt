環境構築手順
---------------------------------------------------------------------------------------------------------------------------------------------
    docker run --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -it --shm-size=4g - v /mnt/disk:/data nvcr.io/nvidia/pytorch:21.03-py3
    apt-get updata
    pip install pillow
    
・detectron2
    git clone https://github.com/facebookresearch/detectron2.git
    pip install --use-feature=2020-resolverblack
    python -m pip install -e detectron2
    apt-get install libgl1-mesa-dev

・unbiased teacher v2
    git clone https://github.com/facebookresearch/unbiased-teacher-v2.git

・BDD100Kのダウンロード
    https://www.vis.xyz/bdd100k/　←ここから100K imagesとDetection2020のlabelをダウンロード
    
・BDD100Kのlabelをcoco形式に変換
    datasets
        -bdd100k
            -images
                -100k
                    -test
                    -train
                    -val
        labels
            -det_20
                -det_train.json
                -det_val.json
        bdd100k2coco.py

　上のディレクトリ構造を作成して以下のコマンドを実行するとcoco形式に変換されたラベルが作成される．
    python bdd2coco.py --bdd_dir ./bdd100k

・dataseedの作成
    dataseed
        -get_bdd100k_supervision.py
　下記のコマンドを実行するとbdd100kのデータシードが作成される
    python get_bdd100k_supervision.py
--------------------------------------------------------------------------------------------------------------------------------------------------

学習コマンド
--------------------------------------------------------------------------------------------------------------------------------------------------
・ラベル付きデータ量1%での学習
    python train_net.py --num-gpus 2 --config configs/Faster-RCNN/bdd100k-standard/faster_rcnn_R50_FPN_ut2_sup1_run0.yaml
・ラベル付きデータ量5%での学習
    python train_net.py --num-gpus 2 --config configs/Faster-RCNN/bdd100k-standard/faster_rcnn_R50_FPN_ut2_sup5_run0.yaml
・ラベル付きデータ量10%での学習
    python train_net.py --num-gpus 2 --config configs/Faster-RCNN/bdd100k-standard/faster_rcnn_R50_FPN_ut2_sup10_run0.yaml
・ラベル付きデータ量100%での学習
    python train_net.py --num-gpus 2 --config configs/Faster-RCNN/bdd100k-standard/faster_rcnn_R50_FPN_ut2_sup100_run0.yaml

・学習済みの重みを読み込んで学習
    python train_net.py --resume --num-gpus 2 --config configs/to/path MODEL.WEIGHTS <your weight>.pth
--------------------------------------------------------------------------------------------------------------------------------------------------

学習のパラメータ設定方法
--------------------------------------------------------------------------------------------------------------------------------------------------
　config内のyamlファイルに以下の項目があるので数値を変えて設定
    BURN_UP_STEP: 30000 ←教師あり学習回数
    MAX_ITER: 120000　←学習回数
    FOW:
      REFERENCE_PERCENT: 1.0 ←基準面積を下位何%にするか
      WEIGHT_ON: True ←FOWを導入するかしないか
--------------------------------------------------------------------------------------------------------------------------------------------------

他データセットで学習する場合
--------------------------------------------------------------------------------------------------------------------------------------------------
scripts/area_process.pyの71, 76行目をクラス数に応じたものに変更し，79行目を削除
scripts/loss_process.pyの10, 11行目を削除
train_net.pyの14, 15行目のパスを適切なものに変更
configs内のyamlファイルのDATASETS, DATALOADERを適切なものに変更
--------------------------------------------------------------------------------------------------------------------------------------------------

その他
--------------------------------------------------------------------------------------------------------------------------------------------------
実験結果等はresultフォルダの中にすべて入っています。