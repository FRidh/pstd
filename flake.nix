{
  description = "PSTD";

  inputs.nixpkgs.url = "github:nixos/nixpkgs?ref=nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
  in rec {

    packages = rec {
      python3 = let
        python = pkgs.python3;
      in python.override {
        packageOverrides = final: prev: {
          pstd = final.callPackage ./default.nix {};
        };
        self = python;
      };

      pstd = python3.pkgs.pstd;
    };

    defaultPackage = packages.pstd;

    # devShell = (pkgs.mkShell {
    #   nativeBuildInputs = with packages.python3.pkgs; [ notebook ] ++ pstd.propagatedBuildInputs;

    # });
    devShell = (packages.python3.withPackages(ps: with ps; [
      notebook
    ] ++ pstd.propagatedBuildInputs)).env;
  });
}
