{
  description = "gambling simulator";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (sys:
      let pkgs = import nixpkgs {
            system = sys;
            overlays = [ (import rust-overlay) ];
          };
          rust = pkgs.rust-bin.stable."1.88.0".default.override {
            extensions = [ "rust-src" "rust-analyzer" ];
          };
          platform = pkgs.makeRustPlatform {
            rustc = rust;
            cargo = rust;
          };
          py = pkgs.python313.withPackages (ps: with ps; [
            matplotlib numpy scipy tqdm
            ipython jupyter ipympl
          ]);
      in rec {
        packages.default = platform.buildRustPackage {
          name = "gambling-simulator";
          src = ./.;
          cargoLock = { lockFile = ./Cargo.lock; };
        };
        devShells.default = pkgs.mkShell {
          packages = [ rust py pkgs.evcxr pkgs.lldb ];
        };
      }
    );
}
