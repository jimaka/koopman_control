"""
PyTorch Dataset wrapper for the Pelican flight data stored in `pelican_dataset_horizontal.npz`.

Each item corresponds to one flight dictionary, preserving the original structure
and fields. Optionally returns tensors for array fields.
"""
from typing import Any, Dict, Optional, List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover - import guarded for environments without torch
    torch = None  # type: ignore

    class Dataset:  # type: ignore
        pass


class PelicanFlightDataset(Dataset):
    """Dataset for Pelican flight data (horizontal plane).

    Args:
        npz_path: Path to the processed dataset (default: 'pelican_dataset_horizontal.npz').
        as_tensors: If True, return torch.Tensor for array fields; otherwise numpy arrays.
        transform: Optional callable applied to each flight dict after conversion.
    """

    def __init__(
        self,
        npz_path: str = "pelican_dataset_horizontal.npz",
        as_tensors: bool = False,
        transform: Optional[Any] = None,
    ) -> None:
        if as_tensors and torch is None:
            raise ImportError(
                "PyTorch is required for as_tensors=True but is not installed."
            )

        self._npz_path = npz_path
        self._as_tensors = as_tensors
        self._transform = transform

        try:
            with np.load(self._npz_path, allow_pickle=True) as data:
                # Stored as an object array of dicts
                flights_obj = data["datas"]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Processed dataset not found at '{self._npz_path}'. "
                f"Run 'python convert_bag_to_npz.py' first."
            ) from e

        # Normalize to a Python list of dicts
        self._flights = [f for f in flights_obj]

    def __len__(self) -> int:
        return len(self._flights)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        flight = self._flights[idx]

        # Shallow copy to avoid mutating the cached dict
        item: Dict[str, Any] = {}
        for k, v in flight.items():
            if isinstance(v, np.ndarray):
                if self._as_tensors:
                    # Convert numeric arrays to torch tensors; keep dtype
                    item[k] = torch.from_numpy(v)  # type: ignore[arg-type]
                else:
                    item[k] = v
            else:
                # Keep scalars (e.g., 'len') as-is
                item[k] = v

        if self._transform is not None:
            item = self._transform(item)
        return item

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        cls = self.__class__.__name__
        return (
            f"{cls}(npz_path='{self._npz_path}', as_tensors={self._as_tensors}, "
            f"num_flights={len(self)})"
        )


class PelicanHorizontalTransitionDataset(Dataset):
    """Transition dataset for horizontal plane motion (3-DOF).

    State vectors are 6D: [x, y, yaw, u, v, r] where:
        - x, y: position in horizontal plane
        - yaw: heading angle (radians)
        - u, v: linear velocities in body frame (surge, sway)
        - r: yaw rate (radians/s)

    Control inputs are 4D: [port_throttle, port_angle, starboard_throttle, starboard_angle]

    Args:
        npz_path: Path to processed NPZ file.
        return_flight_index: If True, also return (flight_idx, t) for testing.
    """

    def __init__(
        self,
        npz_path: str = "pelican_dataset_horizontal.npz",
        return_flight_index: bool = False,
        use_normalized: bool = False,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for PelicanHorizontalTransitionDataset.")

        # Reuse loader from flight dataset
        base = PelicanFlightDataset(npz_path=npz_path, as_tensors=False)
        self._flights: List[Dict[str, Any]] = [base[i] for i in range(len(base))]
        self._index: List[Tuple[int, int]] = []  # (flight_idx, t)
        self._return_flight_index = return_flight_index
        self._use_normalized = use_normalized
        
        # Store normalization statistics if provided
        self._norm_stats = norm_stats
        if norm_stats is None and use_normalized:
            # Compute normalization statistics from training data
            self._compute_normalization_stats()
        
        for fi, flight in enumerate(self._flights):
            # Use the declared flight length to define transitions
            if "len" not in flight:
                continue
            L = int(flight["len"])
            if L <= 1:
                continue
            # Ensure control exists and is long enough for indexing at t
            u = flight.get("Thrusters_CMD")
            if u is None or u.shape[1] < L:
                continue
            for t in range(L - 1):
                self._index.append((fi, t))
                
        print(f"Created HorizontalTransitionDataset with {len(self)} transitions")

    def _compute_normalization_stats(self):
        """Compute normalization statistics from all data."""
        all_states = []
        all_controls = []
        
        for flight in self._flights:
            if "len" not in flight:
                continue
            L = int(flight["len"])
            if L <= 0:
                continue
                
            # Extract states: [x, y, yaw, u, v, r]
            states = self._build_state_vector(flight, None, return_array=True)
            all_states.append(states)
            
            # Extract controls
            controls = flight["Thrusters_CMD"].T  # (N, 4)
            all_controls.append(controls)
        
        if all_states:
            all_states = np.concatenate(all_states, axis=0)
            state_mean = all_states.mean(axis=0, keepdims=True)
            state_std = all_states.std(axis=0, keepdims=True)
            state_std[state_std < 1e-8] = 1.0
        else:
            state_mean = np.zeros((1, 6))
            state_std = np.ones((1, 6))
            
        if all_controls:
            all_controls = np.concatenate(all_controls, axis=0)
            control_mean = all_controls.mean(axis=0, keepdims=True)
            control_std = all_controls.std(axis=0, keepdims=True)
            control_std[control_std < 1e-8] = 1.0
        else:
            control_mean = np.zeros((1, 4))
            control_std = np.ones((1, 4))
        
        self._norm_stats = {
            'state_mean': state_mean,
            'state_std': state_std,
            'control_mean': control_mean,
            'control_std': control_std
        }
        
        print(f"Computed normalization statistics:")
        print(f"  State mean: {state_mean.flatten()}")
        print(f"  State std: {state_std.flatten()}")
        print(f"  Control mean: {control_mean.flatten()}")
        print(f"  Control std: {control_std.flatten()}")

    def _build_state_vector(self, flight: Dict[str, Any], t: Optional[int] = None, 
                          return_array: bool = False) -> np.ndarray:
        """Build state vector from flight data.
        
        Args:
            flight: Flight data dictionary
            t: Time index (if None, returns all states)
            return_array: If True, return (N, 6) array; else return single state
        """
        # Extract components
        pos = flight["Pos"].T  # (2, N) -> (N, 2)
        vel = flight["Vel"].T  # (2, N) -> (N, 2)
        euler = flight["Euler"].T  # (3, N) -> (N, 3)
        pqr = flight["pqr"].T  # (1, N) -> (N, 1)
        
        # Build state vector [x, y, yaw, u, v, r]
        # Note: euler[:, 2] is yaw, pqr[:, 0] is r
        states = np.concatenate([
            pos,  # x, y
            euler[:, 2:3],  # yaw
            vel,  # u, v
            pqr[:, 0:1]  # r
        ], axis=1)  # (N, 6)
        
        if t is not None:
            return states[t]
        return states

    def __len__(self) -> int:
        return len(self._index)

    def _state_at(self, flight: Dict[str, Any], t: int, next_state: bool = False) -> np.ndarray:
        """Get state at time t or t+1."""
        step = t + 1 if next_state else t
        state = self._build_state_vector(flight, step)
        
        # Apply normalization if requested
        if self._use_normalized and self._norm_stats is not None:
            state = (state - self._norm_stats['state_mean'].flatten()) / self._norm_stats['state_std'].flatten()
        
        return state.astype(np.float32)

    def __getitem__(self, idx: int):
        fi, t = self._index[idx]
        flight = self._flights[fi]
        u_arr = flight.get("Thrusters_CMD")
        
        x_t = self._state_at(flight, t, next_state=False)
        x_tp1 = self._state_at(flight, t, next_state=True)
        u_t = u_arr[:, t].astype(np.float32)
        
        # Apply normalization to control if requested
        if self._use_normalized and self._norm_stats is not None:
            u_t = (u_t - self._norm_stats['control_mean'].flatten()) / self._norm_stats['control_std'].flatten()
        
        x_t = torch.from_numpy(x_t)
        x_tp1 = torch.from_numpy(x_tp1)
        u_t = torch.from_numpy(u_t)

        if self._return_flight_index:
            return x_t, x_tp1, u_t, fi, t
        return x_t, x_tp1, u_t
    
    def get_normalization_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Get normalization statistics."""
        return self._norm_stats
    
    def get_data_dimensions(self) -> Dict[str, int]:
        """Get data dimensions."""
        return {
            'state_dim': 6,  # [x, y, yaw, u, v, r]
            'control_dim': 4,  # [port_throttle, port_angle, starboard_throttle, starboard_angle]
        }


class PelicanSequenceDataset(Dataset):
    """Dataset for sequence prediction in horizontal plane motion.
    
    Returns sequences of (states, controls) and future states for prediction.
    
    Args:
        npz_path: Path to processed NPZ file.
        seq_len: Length of input sequence.
        pred_len: Length of prediction sequence.
        stride: Stride for sliding window.
        use_normalized: Whether to use normalized data.
        norm_stats: Precomputed normalization statistics.
    """
    
    def __init__(
        self,
        npz_path: str = "pelican_dataset_horizontal.npz",
        seq_len: int = 50,
        pred_len: int = 10,
        stride: int = 10,
        use_normalized: bool = False,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for PelicanSequenceDataset.")
            
        # Load flight data
        base = PelicanFlightDataset(npz_path=npz_path, as_tensors=False)
        self._flights: List[Dict[str, Any]] = [base[i] for i in range(len(base))]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self._use_normalized = use_normalized
        self._norm_stats = norm_stats
        
        # Build sequence indices
        self._indices: List[Tuple[int, int]] = []  # (flight_idx, start_idx)
        
        for fi, flight in enumerate(self._flights):
            if "len" not in flight:
                continue
            L = int(flight["len"])
            
            # Need enough data for sequence + prediction
            if L < seq_len + pred_len:
                continue
                
            # Generate sliding windows
            for start_idx in range(0, L - seq_len - pred_len, stride):
                self._indices.append((fi, start_idx))
                
        print(f"Created SequenceDataset with {len(self)} sequences (seq_len={seq_len}, pred_len={pred_len})")
        
        # Compute normalization stats if needed
        if norm_stats is None and use_normalized:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics."""
        transition_ds = PelicanHorizontalTransitionDataset(
            npz_path=self._npz_path if hasattr(self, '_npz_path') else "pelican_dataset_horizontal.npz",
            use_normalized=False
        )
        self._norm_stats = transition_ds.get_normalization_stats()
    
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx: int):
        fi, start_idx = self._indices[idx]
        flight = self._flights[fi]
        
        end_idx = start_idx + self.seq_len
        pred_end_idx = end_idx + self.pred_len
        
        # Extract states and controls
        states_all = self._build_state_vector(flight)
        controls_all = flight["Thrusters_CMD"].T  # (N, 4)
        
        # Extract sequences
        input_states = states_all[start_idx:end_idx]  # (seq_len, 6)
        input_controls = controls_all[start_idx:end_idx]  # (seq_len, 4)
        target_states = states_all[end_idx:pred_end_idx]  # (pred_len, 6)
        
        # Apply normalization if requested
        if self._use_normalized and self._norm_stats is not None:
            input_states = (input_states - self._norm_stats['state_mean']) / self._norm_stats['state_std']
            input_controls = (input_controls - self._norm_stats['control_mean']) / self._norm_stats['control_std']
            target_states = (target_states - self._norm_stats['state_mean']) / self._norm_stats['state_std']
        
        # Convert to torch tensors
        input_states = torch.from_numpy(input_states.astype(np.float32))
        input_controls = torch.from_numpy(input_controls.astype(np.float32))
        target_states = torch.from_numpy(target_states.astype(np.float32))
        
        return input_states, input_controls, target_states
    
    def _build_state_vector(self, flight: Dict[str, Any]) -> np.ndarray:
        """Build state vector from flight data."""
        # This is the same as in PelicanHorizontalTransitionDataset
        pos = flight["Pos"].T  # (2, N) -> (N, 2)
        vel = flight["Vel"].T  # (2, N) -> (N, 2)
        euler = flight["Euler"].T  # (3, N) -> (N, 3)
        pqr = flight["pqr"].T  # (1, N) -> (N, 1)
        
        states = np.concatenate([
            pos,  # x, y
            euler[:, 2:3],  # yaw
            vel,  # u, v
            pqr[:, 0:1]  # r
        ], axis=1)  # (N, 6)
        
        return states


def make_transition_dataloader(
    npz_path: str = "pelican_dataset_horizontal.npz",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: Optional[Any] = None,
    use_normalized: bool = False,
):
    """Create a DataLoader for horizontal plane transitions.

    Returns a DataLoader over (x_t, x_{t+1}, u_t) tensors.
    """
    if torch is None:
        raise ImportError("PyTorch is required for the dataloader.")
    
    # First compute normalization stats if needed
    norm_stats = None
    if use_normalized:
        # Create a temporary dataset to compute stats
        temp_ds = PelicanHorizontalTransitionDataset(
            npz_path=npz_path,
            use_normalized=False,
            return_flight_index=False
        )
        norm_stats = temp_ds.get_normalization_stats()
    
    ds = PelicanHorizontalTransitionDataset(
        npz_path=npz_path,
        return_flight_index=False,
        use_normalized=use_normalized,
        norm_stats=norm_stats
    )

    if device is None:
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def collate(batch):
        xs, ys, us = zip(*batch)
        xb = torch.stack(xs, dim=0)
        yb = torch.stack(ys, dim=0)
        ub = torch.stack(us, dim=0)
        return (
            xb.to(device, non_blocking=True),
            yb.to(device, non_blocking=True),
            ub.to(device, non_blocking=True),
        )

    # If moving to a non-CPU device inside collate, disable pin_memory
    effective_pin_memory = False if getattr(device, "type", None) != "cpu" else pin_memory

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        collate_fn=collate,
    )


def make_sequence_dataloader(
    npz_path: str = "pelican_dataset_horizontal.npz",
    seq_len: int = 50,
    pred_len: int = 10,
    stride: int = 10,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: Optional[Any] = None,
    use_normalized: bool = False,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    """Create DataLoader for sequence prediction.
    
    Args:
        npz_path: Path to data file
        seq_len: Input sequence length
        pred_len: Prediction sequence length
        stride: Stride for sliding window
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        device: Target device
        use_normalized: Whether to use normalized data
        split_ratios: Train, validation, test split ratios
        
    Returns:
        train_loader, val_loader, test_loader, norm_stats
    """
    if torch is None:
        raise ImportError("PyTorch is required for the dataloader.")
    
    # First compute normalization stats from training data
    norm_stats = None
    if use_normalized:
        # Load all data to compute stats
        base_ds = PelicanFlightDataset(npz_path=npz_path, as_tensors=False)
        all_flights = [base_ds[i] for i in range(len(base_ds))]
        
        # Collect all states and controls
        all_states = []
        all_controls = []
        
        for flight in all_flights:
            if "len" not in flight:
                continue
            L = int(flight["len"])
            if L <= 0:
                continue
                
            # Build state vector
            pos = flight["Pos"].T
            vel = flight["Vel"].T
            euler = flight["Euler"].T
            pqr = flight["pqr"].T
            
            states = np.concatenate([
                pos,
                euler[:, 2:3],
                vel,
                pqr[:, 0:1]
            ], axis=1)
            
            controls = flight["Thrusters_CMD"].T
            
            all_states.append(states)
            all_controls.append(controls)
        
        if all_states:
            all_states = np.concatenate(all_states, axis=0)
            state_mean = all_states.mean(axis=0, keepdims=True)
            state_std = all_states.std(axis=0, keepdims=True)
            state_std[state_std < 1e-8] = 1.0
        else:
            state_mean = np.zeros((1, 6))
            state_std = np.ones((1, 6))
            
        if all_controls:
            all_controls = np.concatenate(all_controls, axis=0)
            control_mean = all_controls.mean(axis=0, keepdims=True)
            control_std = all_controls.std(axis=0, keepdims=True)
            control_std[control_std < 1e-8] = 1.0
        else:
            control_mean = np.zeros((1, 4))
            control_std = np.ones((1, 4))
        
        norm_stats = {
            'state_mean': state_mean,
            'state_std': state_std,
            'control_mean': control_mean,
            'control_std': control_std
        }
    
    # Create full dataset
    full_ds = PelicanSequenceDataset(
        npz_path=npz_path,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        use_normalized=use_normalized,
        norm_stats=norm_stats
    )
    
    # Split into train, val, test
    n_total = len(full_ds)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    n_test = n_total - n_train - n_val
    
    # Generate random indices
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Create subset datasets
    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_ds, val_indices)
    test_ds = torch.utils.data.Subset(full_ds, test_indices)
    
    # Helper function to create dataloader
    def create_loader(dataset, shuffle_flag):
        if device is None:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        
        def collate_fn(batch):
            input_states, input_controls, target_states = zip(*batch)
            input_states = torch.stack(input_states, dim=0)
            input_controls = torch.stack(input_controls, dim=0)
            target_states = torch.stack(target_states, dim=0)
            
            return (
                input_states.to(device, non_blocking=True),
                input_controls.to(device, non_blocking=True),
                target_states.to(device, non_blocking=True),
            )
        
        effective_pin_memory = False if getattr(device, "type", None) != "cpu" else pin_memory
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
            collate_fn=collate_fn,
        )
    
    train_loader = create_loader(train_ds, shuffle)
    val_loader = create_loader(val_ds, False)
    test_loader = create_loader(test_ds, False)
    
    print(f"Sequence dataloaders created:")
    print(f"  Train: {len(train_ds)} sequences")
    print(f"  Validation: {len(val_ds)} sequences")
    print(f"  Test: {len(test_ds)} sequences")
    
    return train_loader, val_loader, test_loader, norm_stats


if __name__ == "__main__":
    # Test the datasets
    import sys
    
    npz_path = "pelican_dataset_horizontal.npz"
    
    # Test PelicanFlightDataset
    print("Testing PelicanFlightDataset...")
    try:
        flight_ds = PelicanFlightDataset(npz_path=npz_path)
        print(f"Number of flights: {len(flight_ds)}")
        
        if len(flight_ds) > 0:
            flight = flight_ds[0]
            print(f"Flight keys: {list(flight.keys())}")
            for k, v in flight.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k}: {v}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run convert_bag_to_npz.py first to create the dataset.")
        sys.exit(1)
    
    # Test PelicanHorizontalTransitionDataset
    print("\nTesting PelicanHorizontalTransitionDataset...")
    try:
        trans_ds = PelicanHorizontalTransitionDataset(
            npz_path=npz_path,
            return_flight_index=False,
            use_normalized=True
        )
        print(f"Number of transitions: {len(trans_ds)}")
        
        if len(trans_ds) > 0:
            x_t, x_tp1, u_t = trans_ds[0]
            print(f"State dimension: {x_t.shape}")
            print(f"Control dimension: {u_t.shape}")
            print(f"Sample state: {x_t}")
            print(f"Sample control: {u_t}")
            
            # Get normalization stats
            stats = trans_ds.get_normalization_stats()
            if stats:
                print(f"State mean: {stats['state_mean'].flatten()}")
                print(f"State std: {stats['state_std'].flatten()}")
    except Exception as e:
        print(f"Error testing transition dataset: {e}")
    
    # Test make_transition_dataloader
    print("\nTesting make_transition_dataloader...")
    try:
        loader = make_transition_dataloader(
            npz_path=npz_path,
            batch_size=16,
            use_normalized=True
        )
        
        for batch_idx, (x_batch, x_next_batch, u_batch) in enumerate(loader):
            print(f"Batch {batch_idx}:")
            print(f"  States: {x_batch.shape}")
            print(f"  Next states: {x_next_batch.shape}")
            print(f"  Controls: {u_batch.shape}")
            break  # Just test first batch
    except Exception as e:
        print(f"Error testing dataloader: {e}")