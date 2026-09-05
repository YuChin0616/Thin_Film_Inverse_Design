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

# 材料種類順序
FIXED_EPS = np.array(
    [3.9, 11.7, 3.9, 11.7, 3.9, 11.7, 3.9, 11.7, 3.9, 11.7],
    dtype=np.float32,
)


# =============================================================================
# 輸入材料結構，計算指定範圍的 Transmittance
# =============================================================================

def calculate_transmittance_spectrum(
    params: np.ndarray,  # 材料的結構
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
        _, T, _ = component.RTA(
            wavelength,
            eps_in,
            eps_out,
        )

        transmittance.append(T)

    return np.asarray(
        transmittance,
        dtype=np.float32,
    )


# =============================================================================
# 輸入厚度，將厚度與指定的材料順序組合，並生成 Transmittance
# =============================================================================

def transmittance_from_thickness(
    d_nm: np.ndarray,
) -> np.ndarray:
    """
    根據固定 10 層 A/B 結構的厚度，
    計算對應的 Transmittance Spectrum。

    Parameters
    ----------
    d_nm : np.ndarray
        10 層薄膜厚度，單位為 nm，shape = (10,)。

    Returns
    -------
    np.ndarray
        200~800 nm 的 Transmittance Spectrum，
        shape = (601,)。
    """

    if d_nm.shape != (NUM_LAYERS,):
        raise ValueError(
            f"d_nm must have shape ({NUM_LAYERS},), "
            f"got {d_nm.shape}"
        )

    # 組合成 TMM 所需格式：
    # [eps1, d1, eps2, d2, ..., eps10, d10]
    params = np.empty(
        NUM_LAYERS * 2,
        dtype=np.float32,
    )

    # 偶數位置放固定材料介電常數
    params[0::2] = FIXED_EPS

    # 奇數位置放厚度，並將 nm 轉換為 m
    params[1::2] = d_nm * 1e-9

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
# 生成隨機厚度（限制為整數）
# =============================================================================

def generate_random_thickness(
    rng: np.random.Generator,
    d_range_nm: Tuple[int, int] = (10, 300),
) -> np.ndarray:
    """
    隨機生成固定 10 層薄膜的厚度。

    每一層厚度皆從指定範圍內均勻隨機取整數，
    預設範圍為 10~300 nm，包含上下界。

    Parameters
    ----------
    rng : np.random.Generator
        NumPy 隨機數生成器。

    d_range_nm : Tuple[int, int]
        厚度範圍 (minimum, maximum)，單位為 nm。

    Returns
    -------
    np.ndarray
        10 層厚度，shape = (10,)。
    """

    d_nm = rng.integers(
        d_range_nm[0],
        d_range_nm[1] + 1,
        size=NUM_LAYERS,
    )

    return d_nm.astype(np.float32)


# =============================================================================
# 隨機產生一組厚度與對應的 Transmittance
# =============================================================================

def _generate_one_sample(
    seed_int: int,
    d_range_nm: Tuple[int, int],
):
    """
    生成一筆隨機厚度與對應的 Transmittance Spectrum。

    此 Function 主要供 multiprocessing 使用。
    """

    try:
        rng = np.random.default_rng(seed_int)

        d_nm = generate_random_thickness(
            rng,
            d_range_nm=d_range_nm,
        )

        transmittance = transmittance_from_thickness(
            d_nm
        )

        return d_nm, transmittance

    except Exception as exc:
        return seed_int, repr(exc)


# =============================================================================
# 生成完整 Dataset
# =============================================================================

def generate_dataset(
    out_dir: str = "IntegerThin",
    total_samples: int = 1000,
    shard_size: int = 100,
    seed: int = 123,
    d_range_nm: Tuple[int, int] = (10, 300),
    dtype: np.dtype = np.float32,
    num_workers: int = 12,
    chunksize: int = 50,
):
    """
    生成固定 10 層 AB 結構的 Dataset。

    每一筆資料包含：

        thickness : (10,)
            10 層厚度，單位為 nm。

        T : (601,)
            200~800 nm 的 Transmittance Spectrum。

    每個 shard 儲存：

        thickness : (N, 10)
        T         : (N, 601)
        wl_nm     : (601,)

    Dataset 會分成多個 .npz 儲存。

    Parameters
    ----------
    out_dir : str
        Dataset 儲存資料夾。

    total_samples : int
        總共生成的資料數量。

    shard_size : int
        每個 .npz 儲存的資料數量。

    seed : int
        隨機種子，使 Dataset 可以重現。

    d_range_nm : Tuple[int, int]
        厚度範圍，包含上下界，預設為 10~300 nm。

    dtype : np.dtype
        Dataset 儲存格式。

    num_workers : int
        multiprocessing 使用的 CPU process 數量。

    chunksize : int
        multiprocessing 每次分配給 worker 的工作數量。
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

    # 儲存 Dataset 的基本資訊
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
        fixed_eps=FIXED_EPS,
        wl_nm=wl_nm,
    )

    # 固定 random seed，使 Dataset 可以重現
    rng = np.random.default_rng(seed)

    # 計算需要多少個 shard
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

        # 為每一筆 sample 準備獨立的 random seed
        sample_seeds = rng.integers(
            0,
            2**32 - 1,
            size=n_samples,
            dtype=np.uint32,
        )

        # 儲存厚度，shape = (N, 10)
        thickness_batch = np.empty(
            (n_samples, NUM_LAYERS),
            dtype=dtype,
        )

        # 儲存 Transmittance，shape = (N, 601)
        transmittance_batch = np.empty(
            (n_samples, len(wl_nm)),
            dtype=dtype,
        )

        # 使用多核心生成這一個 shard
        with ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:

            results = executor.map(
                _generate_one_sample,
                sample_seeds,
                [d_range_nm] * n_samples,
                chunksize=chunksize,
            )

            # 將每筆結果放入 batch
            for i, result in enumerate(results):

                # 如果 worker 發生錯誤
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

                d_nm, transmittance = result

                thickness_batch[i] = d_nm
                transmittance_batch[i] = transmittance

        # 維持原本的 Dataset 檔名格式
        shard_path = os.path.join(
            out_dir,
            f"Data_{shard_idx:04d}.npz",
        )

        # 保留原本 training notebook 使用的 key：
        # thickness、T、wl_nm
        np.savez_compressed(
            shard_path,
            thickness=thickness_batch,
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
        out_dir="IntegerData1M",
        total_samples=1_000_000,
        shard_size=10_000,
        seed=456,
        d_range_nm=(10, 300),
        dtype=np.float32,
        num_workers=12,
        chunksize=100,
    )