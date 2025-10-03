//
//  TripoSRMlxApp.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/17.
//

import SwiftUI

import MLX
import MLXNN

@main
struct TripoSRMlxApp: App {

    init() {
        // Run model mapping check automatically at app startup
        ModelMappingChecker.runStartupCheck()

        // Run debug check for spherical cameras and ray intersections
//        debugSphericalCamerasAndIntersections()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
