{
  description = "K-space PSTD for Python.";

  inputs.nixpkgs.url = "nixpkgs/nixpkgs-unstable";
  inputs.utils.url = "github:numtide/flake-utils";
  inputs.turbulence.url = "github:FRidh/turbulence";

  outputs = { self, nixpkgs, utils, turbulence }: {
    overlay = final: prev: {
      pythonPackagesOverrides = (prev.pythonPackagesOverrides or []) ++ [
        (self: super: {
          pstd = self.callPackage ./. {};
        })
      ];
      # Remove when https://github.com/NixOS/nixpkgs/pull/91850 is fixed.
      python3 = let
        composeOverlays = nixpkgs.lib.foldl' nixpkgs.lib.composeExtensions (self: super: {});
        self = prev.python3.override {
          inherit self;
          packageOverrides = composeOverlays final.pythonPackagesOverrides;
        };
      in self;
    };
  } // (utils.lib.eachSystem [ "x86_64-linux" ] (system: let
    pkgs = (import nixpkgs {
      inherit system;
      overlays = [ self.overlay turbulence.overlay ];
    });
    python = pkgs.python3;
    pstd = python.pkgs.pstd;
    devEnv = python.withPackages(ps: with ps.pstd; nativeBuildInputs ++ propagatedBuildInputs);
  in {
    # Our own overlay does not get applied to nixpkgs because that would lead to
    # an infinite recursion. Therefore, we need to import nixpkgs and apply it ourselves.
    defaultPackage = pstd;

    devShell = pkgs.mkShell {
      nativeBuildInputs = [
        devEnv
      ];
      shellHook = ''
        export PYTHONPATH=$(readlink -f $(find . -maxdepth 1  -type d ) | tr '\n' ':'):$PYTHONPATH
      '';
    };
  }));
}
