import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np

from simulation import Certain_Wavelength


# =============================================================================
# 參數設定
# =============================================================================

NUM_LAYERS = 10
WAVELENGTH_MIN_NM = 200
WAVELENGTH_MAX_NM = 800
WAVELENGTH_STEP_NM = 1


# =============================================================================
# 輸入材料結構，計算指定範圍的 Transmittance
# =============================================================================

def calculate_transmittance_spectrum(
    params: np.ndarray,
    eps_in: float = 1.0,
    eps_out: float = 1.0,
    wavelength_low: float = 200e-9,
    wavelength_high: float = 800e-9,
) -> np.ndarray:
    """
    輸入完整薄膜結構，計算指定波長範圍的 Transmittance Spectrum。

    params 格式：
    [eps1, d1, eps2, d2, ..., epsN, dN]

    厚度單位為公尺 (m)。

    Returns
    -------
    np.ndarray
        200~800 nm 的 Transmittance Spectrum，共 601 點。
    """

    component = Certain_Wavelength(
        eps_in,
        eps_out,
        params,
    )

    wavelengths = np.arange(
        wavelength_low,
        wavelength_high + 1e-12,
        1e-9,
    )

    transmittance = []

    for wavelength in wavelengths:
        _, T, _ = component.RTA(wavelength, eps_in, eps_out,)
        transmittance.append(T)

    return np.asarray(transmittance, dtype=np.float32,)


# =============================================================================
# 隨機生成材料結構
# =============================================================================

def generate_random_structure(
    rng: np.random.Generator,
    d_range_nm: Tuple[float, float] = (10.0, 300.0),
) -> np.ndarray:
    """
    隨機生成 2~10 層薄膜結構。

    生成順序：
    1. 隨機決定層數。
    2. 隨機決定第一層材料為 eps=3.9 或 eps=11.7。
    3. 後續材料以 ABAB... 方式交錯。
    4. 隨機生成每一層厚度。
    5. 不足 10 層的位置以 eps=11.7、d=0 nm 補齊。

    層數機率：
        2~7 層：每種機率 1/12，合計 50%。
        8~10 層：每種機率 1/6，合計 50%。

    厚度：
        只生成偶數整數：
        10, 12, 14, ..., 300 nm。

    Returns
    -------
    np.ndarray
        shape = (20,)

        格式：
        [eps1, d1_nm, eps2, d2_nm, ..., eps10, d10_nm]
    """

    # 隨機決定層數
    num_layers = rng.choice(
        np.arange(2, 11),
        p=[
            1 / 12,   # 2 layers
            1 / 12,   # 3 layers
            1 / 12,   # 4 layers
            1 / 12,   # 5 layers
            1 / 12,   # 6 layers
            1 / 12,   # 7 layers
            1 / 6,    # 8 layers
            1 / 6,    # 9 layers
            1 / 6,    # 10 layers
        ],
    )

    # 隨機決定第一層材料
    start_eps = rng.choice([3.9, 11.7])

    if start_eps == 3.9:
        other_eps = 11.7
    else:
        other_eps = 3.9

    # 產生 ABAB... 交錯材料
    active_eps = np.empty(
        num_layers,
        dtype=np.float32,
    )

    active_eps[0::2] = start_eps
    active_eps[1::2] = other_eps

    # 將厚度範圍除以 2，
    # 隨機生成整數後再乘以 2，
    # 因此厚度只會是 10, 12, ..., 300 nm
    half_min = int(
        np.ceil(d_range_nm[0] / 2.0)
    )

    half_max = int(
        np.floor(d_range_nm[1] / 2.0)
    )

    active_d_nm = (
        rng.integers(
            half_min,
            half_max + 1,
            size=num_layers,
        )
        * 2
    ).astype(np.float32)

    # 建立完整 10 層陣列
    # 不足 10 層的位置使用 eps=11.7、d=0 nm 補齊
    full_eps = np.full(
        NUM_LAYERS,
        11.7,
        dtype=np.float32,
    )

    full_d_nm = np.zeros(
        NUM_LAYERS,
        dtype=np.float32,
    )

    # 將真正存在的薄膜層放入前面
    full_eps[:num_layers] = active_eps
    full_d_nm[:num_layers] = active_d_nm

    # 組合成：
    # [eps1, d1, eps2, d2, ..., eps10, d10]
    params_nm = np.empty(
        NUM_LAYERS * 2,
        dtype=np.float32,
    )

    params_nm[0::2] = full_eps
    params_nm[1::2] = full_d_nm

    return params_nm


# =============================================================================
# 輸入材料結構，轉換單位後生成 Transmittance
# =============================================================================

def transmittance_from_structure(
    params_nm: np.ndarray,
) -> np.ndarray:
    """
    根據薄膜材料與厚度結構，計算 Transmittance Spectrum。

    Parameters
    ----------
    params_nm : np.ndarray
        shape = (20,)

        格式：
        [eps1, d1_nm, eps2, d2_nm, ..., eps10, d10_nm]

    Returns
    -------
    np.ndarray
        200~800 nm 的 Transmittance Spectrum，
        shape = (601,)。
    """

    if params_nm.shape != (NUM_LAYERS * 2,):
        raise ValueError(
            f"params_nm must have shape ({NUM_LAYERS * 2},), "
            f"got {params_nm.shape}"
        )

    # 複製一份，避免修改原本的結構資料
    params = params_nm.copy()

    # 厚度由 nm 轉換成 m
    params[1::2] *= 1e-9

    # 計算 Transmittance Spectrum
    transmittance = calculate_transmittance_spectrum(
        params
    )

    # 確認輸出維度
    if transmittance.shape != (601,):
        raise ValueError(
            f"Transmittance must have shape (601,), "
            f"got {transmittance.shape}"
        )

    # 確認沒有 NaN 或 Inf
    if not np.isfinite(transmittance).all():
        raise ValueError(
            "Transmittance contains NaN or Inf."
        )

    return transmittance


# =============================================================================
# 隨機產生一組結構與對應的 Transmittance
# =============================================================================

def _generate_one_sample(
    seed_int: int,
    d_range_nm: Tuple[float, float],
):
    """
    生成一筆隨機薄膜結構與對應的 Transmittance Spectrum。

    此 Function 主要供 multiprocessing 使用。
    """

    try:
        rng = np.random.default_rng(seed_int)

        params_nm = generate_random_structure(
            rng,
            d_range_nm=d_range_nm,
        )

        transmittance = transmittance_from_structure(
            params_nm
        )

        return params_nm, transmittance

    except Exception as exc:
        return seed_int, repr(exc)


# =============================================================================
# 生成完整 Dataset
# =============================================================================

def generate_dataset(
    out_dir: str = "IntegerData_VarLayers",
    total_samples: int = 1000,
    shard_size: int = 100,
    seed: int = 123,
    d_range_nm: Tuple[float, float] = (10.0, 300.0),
    dtype: np.dtype = np.float32,
    num_workers: int = 12,
    chunksize: int = 50,
):
    """
    生成隨機 2~10 層薄膜 Dataset。

    每一筆資料包含：

        params : (20,)
            [eps1, d1_nm, eps2, d2_nm, ..., eps10, d10_nm]

        T : (601,)
            200~800 nm 的 Transmittance Spectrum。

    每個 shard 儲存：

        params : (N, 20)
        T      : (N, 601)
        wl_nm  : (601,)

    Dataset 會分成多個 .npz 儲存。
    """

    # 建立 Dataset 資料夾
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    # 波長：200, 201, ..., 800 nm，共 601 點
    wl_nm = np.arange(
        WAVELENGTH_MIN_NM,
        WAVELENGTH_MAX_NM + 1,
        WAVELENGTH_STEP_NM,
        dtype=dtype,
    )

    # 儲存 Dataset 基本資訊
    np.savez(
        os.path.join(
            out_dir,
            "meta.npz",
        ),
        total_samples=np.array(total_samples),
        shard_size=np.array(shard_size),
        seed=np.array(seed),
        d_range_nm=np.array(
            d_range_nm,
            dtype=dtype,
        ),
        wl_nm=wl_nm,
    )

    # 固定 random seed，使 Dataset 可以重現
    rng = np.random.default_rng(seed)

    # 計算總共需要多少個 shard
    num_shards = (
        total_samples + shard_size - 1
    ) // shard_size

    written = 0

    # 一個 shard 一個 shard 生成
    for shard_idx in range(num_shards):

        # 最後一個 shard 可能不足 shard_size
        n_samples = min(
            shard_size,
            total_samples - written,
        )

        # 為每一筆 sample 產生獨立的 random seed
        sample_seeds = rng.integers(
            0,
            2**32 - 1,
            size=n_samples,
            dtype=np.uint32,
        )

        # 儲存結構，shape = (N, 20)
        params_batch = np.empty(
            (n_samples, NUM_LAYERS * 2),
            dtype=dtype,
        )

        # 儲存 Transmittance，shape = (N, 601)
        transmittance_batch = np.empty(
            (n_samples, len(wl_nm)),
            dtype=dtype,
        )

        # 使用多核心生成此 shard
        with ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:

            results = executor.map(
                _generate_one_sample,
                sample_seeds,
                [d_range_nm] * n_samples,
                chunksize=chunksize,
            )

            # 將結果存入 batch
            for i, result in enumerate(results):

                # worker 發生錯誤時，
                # _generate_one_sample 會回傳
                # (seed, error message)
                if isinstance(
                    result[0],
                    (int, np.integer),
                ):
                    print(
                        "[WORKER ERROR]",
                        result,
                    )

                    raise RuntimeError(
                        "Dataset generation failed."
                    )

                params_nm, transmittance = result

                params_batch[i] = params_nm
                transmittance_batch[i] = transmittance

        # 保留原本 variable-layer Dataset 的檔名格式
        shard_path = os.path.join(
            out_dir,
            f"Data_{shard_idx:04d}.npz",
        )

        # 保留原本 training notebook 使用的 key
        np.savez_compressed(
            shard_path,
            params=params_batch,
            T=transmittance_batch,
            wl_nm=wl_nm,
        )

        # 顯示目前生成進度
        written += n_samples

        print(
            f"[OK] wrote {shard_path} "
            f"({written}/{total_samples})"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    mp.freeze_support()
    mp.set_start_method(
        "spawn",
        force=True,
    )

    generate_dataset(
        out_dir="IntegerData_VarLayers",
        total_samples=3_000_000,
        shard_size=10_000,
        seed=456,
        d_range_nm=(10.0, 300.0),
        dtype=np.float32,
        num_workers=12,
        chunksize=100,
    )