//go:build mlx

// vae_types.go - VAE Konfigurationstypen und Datenstrukturen
// Dieses Modul enthält alle grundlegenden Type-Definitionen für den VAE Decoder.

package zimage

// VAEConfig holds VAE decoder configuration
type VAEConfig struct {
	InChannels       int32   `json:"in_channels"`
	OutChannels      int32   `json:"out_channels"`
	LatentChannels   int32   `json:"latent_channels"`
	BlockOutChannels []int32 `json:"block_out_channels"`
	LayersPerBlock   int32   `json:"layers_per_block"`
	NormNumGroups    int32   `json:"norm_num_groups"`
	ScalingFactor    float32 `json:"scaling_factor"`
	ShiftFactor      float32 `json:"shift_factor"`
}
